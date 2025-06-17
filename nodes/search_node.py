from tools.search_tool import web_search_tool
from langchain.schema import Document

def web_search(state):
    question = state["question"]
    documents = state["documents"]
    evaluator = state['retrieval_evaluator']
    dataloader = state['dataloader']
    llm = state['llm']
    websearch_tool = web_search_tool(str(state['dataloader']))
    results = websearch_tool.load_results()
    related_results = []
    results['questions'] = dataloader.input_test_data
    for i , q in enumerate(results['questions']):
        if q == question:
            related_results= results['output_results'][i]
            break
    # Here we should select only relevants using our evaluator
    filtered_docs, _ = evaluator.evaluate_websearch(state["question"], related_results)
    print("Num of Websearch docs: ", len(filtered_docs))
    dataloader.websearch_count += 1
    for d in filtered_docs:
        documents.append(Document(page_content=d, metadata={"source": "Web Search"}))
    return {"documents": documents, "question": question,'dataloader' : dataloader, 'retrieval_evaluator' : evaluator, 'llm': llm}
