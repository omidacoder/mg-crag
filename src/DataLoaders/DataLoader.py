from abc import ABC, abstractmethod
import spacy
import pickle
from tqdm import tqdm
class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> list: # load data as (input,output) pairs
        pass
    
    @abstractmethod
    def get_llm_prompt_template(self,template='default') -> str:
        pass
    
    @abstractmethod
    # save outputs in ram here
    def evaluate_rag_chain(self, rag_chain) -> float: # rag_chain is a function that takes input and gives accuracy/factscore output
        pass
    
    @abstractmethod
    #save outputs in file here. should be called after evaluate_rag_chains
    def save_model_outputs(self, save_path : str='/model_outputs.txt') -> bool:
        pass
    @abstractmethod
    def load_data_for_test(self,num_data: int = 50):
        pass
    
    def strip_contexts(self):
        self.nlp = spacy.load('en_core_web_lg')
        sum_len = 0
        total_count = 0
        for i in tqdm(range(len(self.contexts)), desc='Stripping...'):
            new_context = []
            for c in self.contexts[i]:
                sents = [str(s) for s in self.nlp(c).sents]
                sum_len += len(sents)
                total_count += 1
                if len(sents) < 4:
                    new_context.append(c)
                    continue
                for s in sents:
                    new_context.append(s)
            self.contexts[i] = new_context
        print("avg_len is: ", sum_len / total_count)
    def load_contexts(self, path='stripped_contexts.pickle'):
        with open(path, 'rb') as handle:
            out_dict = pickle.load(handle)
            self.contexts = out_dict['contexts']
    def save_contexts(self, path='stripped_contexts.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump({'contexts': self.contexts}, handle)
                    
