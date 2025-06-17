import os
from tqdm import tqdm
import openai
from time import sleep
import requests
from bs4 import BeautifulSoup
import json
from src.Helpers.Utils import load_model_from_hf
from transformers.pipelines.pt_utils import KeyDataset
import pickle
from datasets import Dataset
class WebSearch:
    def __init__(self,specific_name,extractor="hf"): #extractor also can be openai - specific name is for saving results in pickle file
        self.search_key = os.environ['search_key']
        self.openai_key=os.environ['openai_key']
        self.extractor=extractor
        self.specific_name=specific_name

    def openai_extract_keywords(self,questions, task):
        TASK_PROMPT = {
            "popqa": "Extract at most three keywords separated by comma from the following dialogues and questions as queries for the web search, including topic background within dialogues and main intent within questions. \n\nquestion: What is Henry Feilden's occupation?\nquery: Henry Feilden, occupation\n\nquestion: In what city was Billy Carlson born?\nquery: city, Billy Carlson, born\n\nquestion: What is the religion of John Gwynn?\nquery: religion of John Gwynn\n\nquestion: What sport does Kiribati men's national basketball team play?\nquery: sport, Kiribati men's national basketball team play\n\nquestion: {question}\nquery: ",
            "pubqa": "Extract at most three keywords separated by comma from the claim as queries to extract the key information. \n\nclaim: A mother revealed to her child in a letter after her death that she had just one eye because she had donated the other to him.\nquery: a mother had one eye, donated the other eye to her child after her death.\n\nclaim: WWE wrestler Ric Flair was declared brain dead on 16 May 2019. \nquery: WWE wrestler Ric Flair, brain dead on 16 May 2019\n\nclaim: Current expenditures could likely cover the estimated costs of Medicare for All.\nquery: Current expenditures could likely cover the estimated costs of Medicare for All.\n\nclaim: Measles outbreak kills more than 1,200 in Madagascar.\nquery: Measles outbreak kills more than 1,200 in Madagascar.\n\nclaim: {question}\nquery:  ",
            "arc_challenge": "Extract at most three keywords separated by comma from the question as queries that can be used for searching to extract the key information. \n\nquestion: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nquery: most likely effect of increase in rotation, a planet rotates faster after a meteorite impact\n\nclaim: Jefferson's class was studying sunflowers. They learned that sunflowers are able to make their own food. Which parts of a sunflower collect most of the sunlight needed to make food? \nquery: which parts of a sunflower collect most of the sunlight needed to make food\n\nquestion: A man climbed to the top of a very high mountain. While on the mountain top, he drank all the water in his plastic water bottle and then put the cover back on. When he returned to camp in the valley, he discovered the the empty bottle had collapsed. Which of the following best explains why this happened?\nquery: best to explain the phenomenon, put the cover back of empty plastic bottle on the mountain top, empty bottle collapse in the valley\n\nquestion: There are two types of modern whales: toothed whales and baleen whales. Baleen whales filter plankton from the water using baleen, plates made of fibrous proteins that grow from the roof of their mouths. The embryos of baleen whales have teeth in their upper jaws. As the embryos develop, the teeth are replaced with baleen. Which of the following conclusions is best supported by this information?\nquery: two types of whales toothed whales and baleen whales, baleen whales filter plankton with baleen, teeth replaced with baleen\n\nquestion: {question}\nquery:",
        }
        assert task in TASK_PROMPT, "Your task is not included in TASK_PROMPT for a few-shot prompt template."
        openai.api_key = self.openai_key
        queries = []
        prompt_template = TASK_PROMPT[task]
        for question in tqdm(questions[:]):
            inputs = prompt_template.format(
                question=question
            )
            messages = [
                {"role": "user", "content": inputs},
            ]

            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    temperature=0.1,
                    messages=messages,
                )
            except openai.error.RateLimitError:
                print('Rate limit error')
                sleep(60)
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        temperature=0.1,
                        messages=messages,
                    )
                except openai.error.RateLimitError:
                    print('Rate limit error')
                    sleep(60)
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        temperature=0.1,
                        messages=messages,
                    )
            results = completion["choices"][0]["message"]["content"]
            queries.append(results)
        return queries

    def hf_extract_keywords(self, questions, task):
        TASK_PROMPT = {
            "popqa": "Extract at most three keywords separated by comma from the following dialogues and questions as queries for the web search, including topic background within dialogues and main intent within questions. \n\nquestion: What is Henry Feilden's occupation?\nquery: Henry Feilden, occupation\n\nquestion: In what city was Billy Carlson born?\nquery: city, Billy Carlson, born\n\nquestion: What is the religion of John Gwynn?\nquery: religion of John Gwynn\n\nquestion: What sport does Kiribati men's national basketball team play?\nquery: sport, Kiribati men's national basketball team play\n\nquestion: {question}\nquery: ",
            "pubqa": "Extract at most three keywords separated by comma from the claim as queries to extract the key information. \n\nclaim: A mother revealed to her child in a letter after her death that she had just one eye because she had donated the other to him.\nquery: a mother had one eye, donated the other eye to her child after her death.\n\nclaim: WWE wrestler Ric Flair was declared brain dead on 16 May 2019. \nquery: WWE wrestler Ric Flair, brain dead on 16 May 2019\n\nclaim: Current expenditures could likely cover the estimated costs of Medicare for All.\nquery: Current expenditures could likely cover the estimated costs of Medicare for All.\n\nclaim: Measles outbreak kills more than 1,200 in Madagascar.\nquery: Measles outbreak kills more than 1,200 in Madagascar.\n\nclaim: {question}\nquery:  ",
            "arc_challenge": "Extract at most three keywords separated by comma from the question as queries that can be used for searching to extract the key information. \n\nquestion: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nquery: most likely effect of increase in rotation, a planet rotates faster after a meteorite impact\n\nclaim: Jefferson's class was studying sunflowers. They learned that sunflowers are able to make their own food. Which parts of a sunflower collect most of the sunlight needed to make food? \nquery: which parts of a sunflower collect most of the sunlight needed to make food\n\nquestion: A man climbed to the top of a very high mountain. While on the mountain top, he drank all the water in his plastic water bottle and then put the cover back on. When he returned to camp in the valley, he discovered the the empty bottle had collapsed. Which of the following best explains why this happened?\nquery: best to explain the phenomenon, put the cover back of empty plastic bottle on the mountain top, empty bottle collapse in the valley\n\nquestion: There are two types of modern whales: toothed whales and baleen whales. Baleen whales filter plankton from the water using baleen, plates made of fibrous proteins that grow from the roof of their mouths. The embryos of baleen whales have teeth in their upper jaws. As the embryos develop, the teeth are replaced with baleen. Which of the following conclusions is best supported by this information?\nquery: two types of whales toothed whales and baleen whales, baleen whales filter plankton with baleen, teeth replaced with baleen\n\nquestion: {question}\nquery:",
        }
        assert task in TASK_PROMPT, "Your task is not included in TASK_PROMPT for a few-shot prompt template."
        queries = []
        prompt_template = TASK_PROMPT[task]
        pipeline = load_model_from_hf(batch_size=10)
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        text = [prompt_template.format(question=question)
                for question in questions]
        if task == 'arc_challenge':
            text = [prompt_template.format(question=question.split("A)")[0].split("1)")[
                                           0].strip()) for question in questions]
        data = {
            'TD': [i for i in range(len(questions))],
            'text': text
        }
        dataset = Dataset.from_dict(data)
        for query in tqdm(pipeline(KeyDataset(dataset, "text"),do_sample=False,
                                   max_new_tokens=50, return_full_text=False)):
            # for pubhealth dataset :
            if task == 'pubqa':                      
                queries.append(query[0]['generated_text'].split("\n\nclaim:")[0].strip())   
            # for other datasets :
            else:                    
                queries.append(query[0]['generated_text'].split("\n\nquestion:")[0].strip())
        self.queries = queries
        return queries

    def generate_knowledge_q(self,questions, task, mode):
        self.questions = questions
        print("Initial questions:", self.questions)  # Debugging line
        if self.extractor == 'hf':
            extract_keywords = self.hf_extract_keywords
        else:
            extract_keywords = self.openai_extract_keywords
        if task == 'bio':
            queries = [q[:] for q in questions]
        else:
            queries = extract_keywords(questions, task)
        if mode == 'wiki':
            search_queries = ["Wikipedia, " + e for e in queries]
        else:
            search_queries = queries
        return search_queries

    def search(self, search_path="None"):
        queries = self.queries
        url = "https://google.serper.dev/search"
        responses = []
        search_results = []
        for query in tqdm(queries[:], desc="Searching for urls..."):
            payload = json.dumps(
                {
                    "q": query
                }
            )
            headers = {
                'X-API-KEY': self.search_key,
                'Content-Type': 'application/json'
            }

            reconnect = 0
            while reconnect < 3:
                try:
                    response = requests.request(
                        "POST", url, headers=headers, data=payload)
                    break
                except (requests.exceptions.RequestException, ValueError):
                    reconnect += 1
                    print('url: {} failed * {}'.format(url, reconnect))
            # result = response.text
            print("The Result is : ", response.text)
            result = json.loads(response.text)
            if "organic" in result:
                results = result["organic"][:10]
            else:
                results = query
            responses.append(results)

            search_dict = [{"queries": query, "results": results}]
            search_results.extend(search_dict)
        if search_path != 'None':
            with open(search_path, 'w') as f:
                output = json.dumps(search_results, indent=4)
                f.write(output)
        self.search_results = search_results
        return search_results


    def test_page_loader(self, url):
        import signal

        def handle(signum, frame):
            raise RuntimeError
        reconnect = 0
        while reconnect < 1:
            try:
                # below lines are runnable only on a unix system
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(180)
                response = requests.get(url)
                break
            except (requests.exceptions.RequestException, ValueError, RuntimeError):
                reconnect += 1
                print('url: {} failed * {}'.format(url, reconnect))
                if reconnect == 1:
                    return []
        try:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
        except:
            return []
        if soup.find('h1') is None or soup.find_all('p') is None:
            return []
        paras = []
        title = soup.find('h1').text
        paragraphs = soup.find_all('p')
        for i, p in enumerate(paragraphs):
            if len(p.text) > 10:
                paras.append(title + ': ' + p.text)
        return paras


    def visit_pages(self, mode='all'): #or mode can be 'wiki'
        questions = self.questions
        web_results = self.search_results
        top_n = 5
        titles = []
        urls = []
        snippets = []
        queries = []
        for i, result in enumerate(web_results[:]):
            title = []
            url = []
            snippet = []
            if type(result["results"]) == list:
                for page in result["results"][:5]:
                    if mode == "wiki":
                        if "wikipedia" in page["link"]:
                            title.append(page["title"])
                            url.append(page["link"])
                    else:
                        title.append(page["title"])
                        url.append(page["link"])
                    if "snippet" in page:
                        snippet.append(page["snippet"])
                    else:
                        snippet.append(page["title"])
            else:
                titles.append([])
                urls.append([])
                snippets.append([result["results"]])
                queries.append(result["queries"])
                continue
            titles.append(title)
            urls.append(url)
            snippets.append(snippet)
            queries.append(result["queries"])
        output_results = []
        progress_bar = tqdm(
            range(len(questions[:])), desc="Visiting page content...")
        assert len(questions) == len(urls), (len(questions), len(urls))
        i = 0
        for title, url, snippet, query in zip(titles[i:], urls[i:], snippets[i:], queries[i:]):
            if url == []:
                results = '; '.join(snippet)
            else:
                strips = []
                for u in url:
                    strips += self.test_page_loader(u)
                if strips == []:
                    output_results.append('; '.join(snippet))
                    results = '; '.join(snippet)
                else:
                    #changed this part from T5 model
                    results = strips
            i += 1
            output_results.append(results)
            progress_bar.update(1)
        self.output_results = output_results
        self.queries = queries
        self.questions = questions
        return output_results
        
    def save_results(self):
        # save results to file
        with open('outputs/search_results/'+self.specific_name + '.pickle', 'wb') as handle:
            pickle.dump({'questions': self.questions, 'queries': self.queries,
                        'output_results': self.output_results}, handle)
        return self.output_results
    def load_results(self):
        with open('outputs/search_results/'+self.specific_name + '.pickle', 'rb') as handle:
            results = pickle.load(handle)
            self.output_results = results['output_results']
            self.queries = results['queries']
            self.questions = results['questions']
            return results
    
    def save_queries(self):
        with open('outputs/search_results/'+self.specific_name + '_queries.pickle', 'wb') as handle:
            pickle.dump({'queries': self.queries}, handle)
    def load_queries(self):
        with open('outputs/search_results/'+self.specific_name + '_queries.pickle', 'rb') as handle:
            results = pickle.load(handle)
            self.queries = results['queries']
            return self.queries
    def save_search_results(self):
        with open('outputs/search_results/'+self.specific_name + '_search_results.pickle', 'wb') as handle:
            pickle.dump({'search_results': self.search_results}, handle)
    def load_search_results(self):
        with open('outputs/search_results/'+self.specific_name + '_search_results.pickle', 'rb') as handle:
            results = pickle.load(handle)
            self.search_results = results['search_results']
            return self.search_results
        
