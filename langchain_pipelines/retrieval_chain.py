from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import os
import streamlit as st

LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]

# Initialize Retrieval Chain
def get_retrieval_chain():
    """
    Create and return a RetrievalQA chain using Chroma and HuggingFace embeddings.

    Returns:
        RetrievalQA: The retrieval-based question-answering chain.
    """
    # Initialize embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Top 5 matches

    # Initialize Hugging Face model via HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_REPO_ID,
        model_kwargs={
            "max_length": 2048,
        },
        huggingfacehub_api_token=HF_TOKEN,
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
