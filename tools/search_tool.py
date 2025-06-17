from src.Helpers.WebSearch import WebSearch

def web_search_tool(specific_name):
    return WebSearch(specific_name=specific_name,extractor='openai')
