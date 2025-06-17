from langchain.schema import Document

def retrieve(state):
    question = state["question"]
    dataloader = state["dataloader"]
    retrieval_evaluator = state['retrieval_evaluator']
    llm = state['llm']
    questions = []
    documents = []
    if str(dataloader) == 'pubqa' or str(dataloader) == 'popqa' or str(dataloader) == 'arc':
        questions = dataloader.input_test_data
    for i , q in enumerate(questions):
        if q == question:
            documents = [Document(page_content=doc, metadata={"source": "Knowledge"}) for doc in dataloader.contexts[i]]
            break
    return {"documents": documents, "question": question,'dataloader' : dataloader,'retrieval_evaluator' : retrieval_evaluator, 'llm': llm}
