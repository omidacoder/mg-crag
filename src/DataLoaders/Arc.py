from src.DataLoaders.DataLoader import DataLoader
from langchain.prompts import PromptTemplate
from src.Helpers.Utils import check_different_shapes_of_choices
from src.Helpers.Utils import process_arc_choices
import json
from tqdm import tqdm
import random
import pickle


class Arc(DataLoader):
    def __str__(self):
        return 'arc'

    def __init__(self, contexts_path=None):
        self.template = 'default'
        self.generation_checkpoint = None
        self.websearch_count = 0
        self.strip = False
        self.contexts_path = contexts_path
        pass
    
    def load_data(self):
        file = open(
            'src/Dataset/eval_data/arc_challenge_processed.jsonl', 'r', encoding='utf-8')
        lines = file.readlines()
        # print(f"Number of lines read: {len(lines)}")
        self.test_data = []
        self.input_test_data = []
        self.output_test_data = []
        self.contexts = []
        for line in lines:
            data = json.loads(line)
            choices = data['choices']
            question = data['question']
            for i, choice in enumerate(choices['label']):
                question += ' ' + process_arc_choices(choice) + ': ' + \
                    choices['text'][i] + ';'
            self.input_test_data.append(question)
            self.output_test_data.append(process_arc_choices(data['answerKey']))
            docs = []
            for doc in data['ctxs']:
                docs.append(doc['text'].strip())
            self.contexts.append(docs)
        assert len(self.contexts) == len(
            self.input_test_data) == len(self.output_test_data)
        if self.strip:
            self.strip_contexts()
        if self.contexts_path:
            self.load_contexts(path=self.contexts_path)
        return list(zip(self.input_test_data, self.output_test_data))

    def train_test_split(self, split_point: float = 0.8 ):
        return None, list(zip(self.input_test_data, self.output_test_data))

    def get_llm_prompt_template(self,template='default'):
        if template == 'default':
            return PromptTemplate(
                template="Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: {context}\n\nInstruction: Given four answer candidates, A, B, C and D, choose the best answer choice.\nQuestion: {question}\n\nAnswer:",
                input_variables=["context", "question"],
            )
        elif template == 'self-rag':
            return PromptTemplate(
                template="### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n### Input:\n{question}\n\n### Response:\n{context}",
                input_variables=["context", "question"],
            )
        else:
            return PromptTemplate(
                # template="### Instruction:\nIs the following statement correct or not? In Response say 'True' if it's correct; otherwise say 'False'.\n\n### Input:\n{question}\n\n### Response:\n{context}",
                template="### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n### Input:\n{question}\n{context}\n\n### Response:",
                input_variables=["context", "question"],
            )
        
    
    def evaluate_rag_chain(self, rag_chain) -> float:
        if self.generation_checkpoint is None:
            self.generation_checkpoint = 0
            self.generations = []
        for input in tqdm(self.input_test_data[self.generation_checkpoint:], "evaluating on arc... " if self.generation_checkpoint == 0 else "continuing evaluation from checkpoint..."):
            response = rag_chain(input.strip().replace(
                '\n', ' '))
            # print("the response of network is: ", response)
            self.generations.append(response.replace('\n', ' ').strip())
            # save results during process
            f = open("ARC_outputs.txt", "a")
            f.write(input.strip().replace('\n', ' ') + '\n')
            f.write(response.strip().replace('\n', ' ') + '\n')
            f.close()
        # calculate accuracy here based on exact match
        total = len(self.generations)
        corrects = 0
        for input, output, generation in zip(self.input_test_data, self.output_test_data, self.generations):
            if check_different_shapes_of_choices(output, generation):
                corrects += 1
        self.accuracy = (corrects / total) * 100
        return self.accuracy
    
    def accuracy(self):
        total = len(self.generations)
        corrects = 0
        for input, output, generation in zip(self.input_test_data, self.output_test_data, self.generations):
            if check_different_shapes_of_choices(output, generation):
                corrects += 1
        self.accuracy = (corrects / total) * 100
        return self.accuracy
    def statistics(self):
        print("Number of websearches: ", self.websearch_count)
        print("Share of websearches: ", str(self.websearch_count * 100 / len(self.generations)) + "%")
        print("Accuracy: ", self.accuracy())
    
    def save_model_outputs(self, save_path: str = 'outputs/popQA_outputs.txt') -> bool:
        with open(save_path, 'wb') as handle:
            pickle.dump({'input_test_data': self.input_test_data, 'generations': self.generations}, handle)

    def load_data_for_test(self,num_data: int = 50):
        random_indices = random.sample(range(len(self.input_test_data)), num_data)
        self.input_test_data = [self.input_test_data[i] for i in random_indices]
        self.output_test_data = [self.output_test_data[i] for i in random_indices]
        self.contexts = [self.contexts[i] for i in random_indices]







