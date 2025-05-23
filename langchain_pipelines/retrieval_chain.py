from langchain_astradb import AstraDBVectorStore
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

# VARIABLES, TOKENS AND KEYS
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
ASTRA_DB_COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION_NAME")

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = AstraDBVectorStore(
    collection_name=ASTRA_DB_COLLECTION_NAME,
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

@tool(response_format="content_and_artifact")
def retriever(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs