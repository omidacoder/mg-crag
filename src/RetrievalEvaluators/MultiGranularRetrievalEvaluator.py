import numpy as np
import random

from transformers import T5ForSequenceClassification
from transformers import T5Tokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from transformers import get_scheduler
from sentence_transformers import SentenceTransformer
import spacy
from langchain.schema import Document
from src.Helpers.ResNet import ResNet
from src.Helpers.Utils import accuracy
from src.RetrievalEvaluators.RetrievalEvaluator import RetrievalEvaluator
from configuration import reranker_model, mechanism

class MultiGranularRetrievalEvaluator(RetrievalEvaluator):

    def __init__(self):
        self.seed = 42
        self.batch_size = 1
        self.num_epochs = 50
        self.LOW_LABEL = 0
        self.AMBIGOUS_LABEL = 1
        self.HIGH_LABEL = 2
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.max_length = 10000000
        self.s_model= SentenceTransformer(reranker_model)

    def fine_tune_model(self, train_texts, train_label, save_path='./outputs/'):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # main code from CRAG article
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model = T5ForSequenceClassification.from_pretrained(
            "t5-large", num_labels=3)
        train_data = tokenizer(train_texts, padding="max_length",
                               max_length=512, truncation=True, return_tensors="pt")
        train = TensorDataset(
            train_data["input_ids"], train_data["attention_mask"], torch.tensor(train_label))
        train_dataloader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True, sampler=None)
        optimizer = Adam(model.parameters(), lr=0.001,
                         betas=(0.9, 0.999), eps=1e-08)
        num_training_steps = self.num_epochs * len(train_dataloader)
        print(num_training_steps)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model.to(self.device)
        for i, epoch in enumerate(range(self.num_epochs)):
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 10 == 0 and not step == 0:
                    print("step: ", step, "  loss:",
                          total_loss/(step*self.batch_size))
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                model.zero_grad()
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs.loss
                loss.mean().backward()
                total_loss += loss.mean().item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            avg_train_loss = total_loss / len(train_dataloader)
            print("avg_loss:", avg_train_loss)
            print("train accuracy: ", accuracy(
                model, tokenizer, train_texts[:188], train_label[:188]))
            model.save_pretrained(save_path + "/ep{}".format(i))
            tokenizer.save_pretrained(save_path + "/ep{}".format(i))
            self.model = model
            self.tokenizer = tokenizer

    def load_pretrained_model(self, plre_load_path='plre.pt', slre_load_path='slre.pt'):
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = SentenceTransformer(
            'sentence-transformers/gtr-t5-large').to('cuda')
        self.proj_net_strips = ResNet(768)
        self.proj_net_strips.load_state_dict(torch.load(slre_load_path))
        self.proj_net =ResNet(768) 
        self.proj_net.load_state_dict(torch.load(plre_load_path))
        return self

    def evaluate_single(self, q: str, doc: str, mode='passage'):
        # we can use few shot to show mid levels
        if not (type(doc) is str):
          doc = doc.page_content
        # prompt_template = "qnli question: Is the statement 'qqqqqq' correct? sentence: cccccc"
        prompt_template = "qnli question: qqqqqq sentence: cccccc"
        input_text = prompt_template.replace(
            'qqqqqq', q).replace('cccccc', doc)
        embeddings = torch.tensor(self.model.encode(input_text))
        if mode == 'passage':
          output = torch.nn.functional.softmax(self.proj_net(embeddings), dim=0)
        if mode == 'strip':
          output = torch.nn.functional.softmax(self.proj_net_strips(embeddings), dim=0)
        _, predicted = torch.max(output, 0)
        return output[self.HIGH_LABEL], predicted
    def evaluate_batch(self, q: str, docs: list):
        # Initialize lists for different categories
        high_docs = []
        medium_docs = []
        high_probs = []
        medium_probs = []
        for d in docs:
            # Evaluate single document
            prob, out = self.evaluate_single(q, d)
            # Categorize documents based on the evaluation output
            if out == self.HIGH_LABEL:
                high_docs.append(d)
                high_probs.append(prob)
            elif out == self.AMBIGOUS_LABEL:
                medium_docs.append(d)
                medium_probs.append(prob)
        high_strips = []
        medium_strips = []
        all_sents = []
        if mechanism == "strict":
            for i,d in enumerate(high_docs):
                all_sents += [str(s) for s in self.nlp(d.page_content).sents]
            for sentence in all_sents:
                prob, out = self.evaluate_single(q, sentence, mode="strip")
                if out == self.HIGH_LABEL:
                    high_strips.append(Document(page_content=sentence, metadata={"source": "Knowledge"}))
                if out == self.AMBIGOUS_LABEL:
                    medium_strips.append(Document(page_content=sentence, metadata={"source": "Knowledge"}))
            all_sents = []
            for i,d in enumerate(medium_docs):
                all_sents += [str(s) for s in self.nlp(d.page_content).sents]
            for sentence in all_sents:
                prob, out = self.evaluate_single(q, sentence, mode="strip")
                if out != self.LOW_LABEL:
                    medium_strips.append(Document(page_content=sentence, metadata={"source": "Knowledge"}))
        elif mechanism == "moderate":
            for i,d in enumerate(high_docs):
                all_sents += [str(s) for s in self.nlp(d.page_content).sents]
            for sentence in all_sents:
                prob, out = self.evaluate_single(q, sentence, mode="strip")
                if out != self.LOW_LABEL:
                    high_strips.append(Document(page_content=sentence, metadata={"source": "Knowledge"}))
            all_sents = []
            for i,d in enumerate(medium_docs):
                all_sents += [str(s) for s in self.nlp(d.page_content).sents]
            for sentence in all_sents:
                prob, out = self.evaluate_single(q, sentence, mode="strip")
                if out != self.LOW_LABEL:
                    medium_strips.append(Document(page_content=sentence, metadata={"source": "Knowledge"}))
        else: #lenient
            for i,d in enumerate(high_docs):
                all_sents += [str(s) for s in self.nlp(d.page_content).sents]
            for sentence in all_sents:
                prob, out = self.evaluate_single(q, sentence, mode="strip")
                if out != self.LOW_LABEL:
                    high_strips.append(Document(page_content=sentence, metadata={"source": "Knowledge"}))
            all_sents = []
            for i,d in enumerate(medium_docs):
                all_sents += [str(s) for s in self.nlp(d.page_content).sents]
            for sentence in all_sents:
                prob, out = self.evaluate_single(q, sentence, mode="strip")
                if out != self.LOW_LABEL:
                    high_strips.append(Document(page_content=sentence, metadata={"source": "Knowledge"}))
        # Check conditions for different outcomes
        high_strips = self.sort_docs(q,high_strips,high_strips)
        medium_strips = self.sort_docs(q,medium_strips,medium_strips)
        # Check conditions for different outcomes
        if len(high_strips) > 0:  # At least one strip is labeled "high"
            return high_strips[:3], "No" if len(high_strips) > 10 else "Yes"
        else:
            return medium_strips[:3], "Yes"
    def sort_docs(self, q, docs):
        def cosine_similarity(A,B):
          return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
        # def standardize(vector):
        #   if len(vector) == 0:
        #     return np.array([])
        #   vector = np.array(vector)
        #   mean = np.mean(vector)
        #   std_dev = np.std(vector)
        #   return (vector - mean) / std_dev
        s_model= self.s_model
        q_emb = s_model.encode(q)
        scores = []
        for d in docs:
          if not (type(d) is str):
            d_emb = s_model.encode(d.page_content)
          else:
            d_emb = s_model.encode(d)
          scores.append(cosine_similarity(q_emb,d_emb))
        avg_scores = [scores[i] for i in range(len(scores))]
        top_n_indices = np.argsort(np.array(avg_scores))[::-1]
        top_corresponding_values = [docs[i] for i in top_n_indices]
        return top_corresponding_values
    def evaluate_websearch(self, q: str, docs: list):
          strips = []
          for d in docs:
              for sentence in self.nlp(d).sents:
                  strips.append(sentence)
          high_docs = []
          high_probs = []
          for d in strips:
              prob, out = self.evaluate_single(q, str(d))
              if out == self.HIGH_LABEL:
                  high_docs.append(str(d))
                  high_probs.append(prob)
          high_docs = self.sort_docs(q,high_docs,high_probs)
          return high_docs[:2], "No"