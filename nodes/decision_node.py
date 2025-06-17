def decide_to_websearch(state):
    web_search = state["web_search"]
    if web_search == "Yes":
        return "web_search_node"
    else:
        return "generate"
