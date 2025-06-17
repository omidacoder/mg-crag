from tools.generate_tool import rag_chain,formatter
# from tools.log_tool import logger

def generate(state):
    question = state["question"]
    documents = state["documents"]
    dataloader = state['dataloader']
    prompt_template = dataloader.get_llm_prompt_template(dataloader.template)
    llm = state['llm']

    # logger.info("final context is: " + format_docs(documents))
    
    chain = rag_chain(prompt_template, llm)
    # input_dict = {"context": format_docs(documents), "question": question}
    # let's get only 2 high docs to see the effect
    
    # print("here is the input_dict : ",input_dict)
    has_exception = True
    while has_exception:
        try:
            context = formatter(documents,dataloader.template)
            input_dict = {"context": context, "question": question}
            generation = chain.invoke(input_dict)
            has_exception = False
        except:
            has_exception = True
            documents.pop()
    print("Number of docs user: ", len(documents))
    print("Here is the Input dict: ", input_dict)
    return {"context": documents, "question": question, "generation": generation}
