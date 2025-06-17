
import pandas as pd
import json
import pickle
class DatasetCreator:
    def __init__(self):
        self.arc_df = pd.read_parquet('src/Dataset/train_data/train_arc.parquet', engine='pyarrow')
        self.pubhealth_df = pd.read_csv(
            'src/Dataset/train_data/train_pubhealth.tsv', sep='\t')
        self.popqa_data = self.read_popqa_data()
    def read_popqa_data(self):
        data = []
        file_path = 'src/Dataset/train_data/train_popqa.txt'
        with open(file_path, 'r',encoding="utf8") as file:
            current_q = ''
            docs = []
            for line in file:
                parts = line.strip().split('[SEP]')
                q = parts[0].strip()
                doc = parts[1][:-1].strip()
                if current_q == q:
                    docs.append({'text':doc})
                else:
                    data.append({'question': current_q, 'contexts': docs})
                    current_q = q
                    docs = []
        print("popqa length is: ",len(data))
        return data[1:]
    def create_jsonl_file(self,filename='outputs/train_data_questions.jsonl'):
        arc_questions = self.arc_df['question'].tolist()
        arc_choices = self.arc_df['choices'].tolist()
        print(arc_choices[0]['text'].tolist())
        labels = ['A) ', 'B) ', 'C) ', 'D) ', 'E) ']
        data = []
        for i in range(len(arc_questions)):
            question = arc_questions[i] + ' '
            for j in range(len(arc_choices[i]['text'].tolist())):
                question += labels[j] + arc_choices[i]['text'].tolist()[j] + ' '
            data.append({"question" : question.strip()})
        print(data[0])
        print(data[1])
        print(data[2])
        # data.extend([{'question': d} for d in self.pubhealth_df['claim'].tolist()])
        # data.extend([{'question' : d} for d in self.popqa_data])
        # Write data to JSONL file
        print("length of data is: ", len(data))
        with open(filename, 'w') as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + '\n')
        print('done!')

    def separate_pubhealth_retrievals(self, retrieval_path="outputs/retrievals_from_arc_pubhealth_train.pickle"):
        pubhealth_data = [d for d in self.pubhealth_df['claim'].tolist()]
        with open(retrieval_path, 'rb') as handle:
            out = pickle.load(handle)
        results = []
        counter = 0
        for o in out:
            flag = False
            for p in pubhealth_data:
                if p == o['question']:
                    flag = True
                    results.append(o)
                    break
            if flag == False:
                counter+=1
        # with open("outputs/retrievals_from_pubhealth_train.pickle", 'wb') as handle:
        #     pickle.dump(results, handle)
    def convert_popqa_to_pickle(self):
        with open("outputs/retrievals_from_popqa_train.pickle", 'wb') as handle:
            pickle.dump(self.popqa_data, handle)
                
        
        
        
