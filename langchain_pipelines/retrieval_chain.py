from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import os
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize Retrieval Chain
def get_retrieval_chain():
    """
    Create and return a RetrievalQA chain using Chroma and OpenAI embeddings.

    Returns:
        RetrievalQA: The retrieval-based question-answering chain.
    """
    # Initialize embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize OpenAI model via ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
    )

    # Create RetrievalQA chain
    retrieval_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return retrieval_chain

# Query the Retrieval Chain
def retrieve_report(query):
    """
    Query the retrieval chain and get results.

    Args:
        query (str): The natural language query to search the vectorstore.

    Returns:
        dict: The result containing the answer and source documents.
    """
    retrieval_chain = get_retrieval_chain()
    result = retrieval_chain(query)
    print("Answer:", result.get("result"))
    return result
