from langchain_astradb import AstraDBVectorStore
import os
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

# VARIABLES, TOKENS AND KEYS
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = AstraDBVectorStore(
    collection_name="urdecontest1",
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs