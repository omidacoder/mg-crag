from langchain_core.output_parsers import StrOutputParser


def rag_chain(prompt_template, llm):
    return prompt_template | llm | StrOutputParser()


def formatter(docs, template):

    def format_docs_stripped(sentences):
        if len(sentences) > 0:
            return "[Retrieval]<paragraph>" + " ".join(sentence.page_content.strip().replace("\n", " ") for sentence in sentences) + "</paragraph>"
        return ""
    def format_docs_stripped_default(sentences):
        if len(sentences) > 0:
            return " ".join(sentence.page_content.strip().replace("\n", " ") for sentence in sentences)
        return ""
    if template == 'default':
        return format_docs_stripped_default(docs)
    else:
        return format_docs_stripped(docs)
