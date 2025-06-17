from abc import ABC, abstractmethod


class RetrievalEvaluator(ABC):
    @abstractmethod
    def evaluate_single(self, q : str, doc : str, mode="passage") -> float: # gives a score in (LowerBound , UpperBound) 
        pass
    @abstractmethod
    def evaluate_batch(self, q : str, docs : list): # returns result docs 
        pass
    
    def evaluate_websearch(self, q : str, docs : list):
        return self.evaluate_batch(q,docs)[0]
    
