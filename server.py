# server.py
from mcp.server.fastmcp import FastMCP
from rag import *

# Create an MCP server
mcp = FastMCP("MCP-RAG-app",
              host="127.0.0.1",
              port=8080,
              )

@mcp.tool()
def machine_learning_faq_retrieval_tool(query: str) -> str:
    """Retrieve the most relevant documents from the machine learning
       FAQ collection. Use this tool when the user asks about ML.

    Input:
        query: str -> The user query to retrieve the most relevant documents

    Output:
        context: str -> most relevant documents retrieved from a vector DB
    """

    # check type of text
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    retriever = Retriever(QdrantVDB("ml_faq_collection"), EmbedData())
    response = retriever.search(query)

    return response


@mcp.tool()
def serper_web_search_tool(query: str) -> list[str]:
    """
    Search for information on a given topic using the Serper API
    (Google Search results via serper.dev).
    Use this tool when the user asks about a specific topic or question
    that is not related to general machine learning.

    Input:
        query: str -> The user query to search for information

    Output:
        context: list[str] -> list of most relevant web search results
    """
    # check type of text
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    import os
    import requests
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY not set in environment")

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Return organic search results (list of dicts: title, link, snippet, etc.)
    organic_results = data.get("organic", [])

    # Return a flat list of readable strings so it matches the tool's
    # declared return type (list[str])
    results = []
    for item in organic_results:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        results.append(f"{title}\n{snippet}\n{link}")

    return results


if __name__ == "__main__":
    print("Starting MCP server at http://127.0.0.1:8080 on port 8080")
    mcp.run(transport="sse")