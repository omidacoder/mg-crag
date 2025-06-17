# from tools.log_tool import logger


def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    evaluator = state['retrieval_evaluator']
    dataloader = state['dataloader']
    llm = state['llm']
    filtered_docs = []
    
    # For Corrective RAG uncomment below line
    filtered_docs , web_search = evaluator.evaluate_batch(question , documents)
    
    # For Simple RAG uncomment below lines
    # filtered_docs= documents[:3]
    # web_search= "No"
    # logger.info("The Question is: " + state["question"])
    # logger.info("Websearch result is: " + web_search)
    # doc_string = ""
    # for d in filtered_docs:
    #     doc_string += d.page_content + '\n\n'
    # logger.info("inner knowledge filtered_docs: " + doc_string)
    # print(f"The websearch result is : {web_search}")
    # print(f"filtered_docs are : {filtered_docs}")
    return {"documents": filtered_docs, "question": question, "web_search": web_search,'dataloader' : dataloader , 'retrieval_evaluator' : evaluator, 'llm': llm}
