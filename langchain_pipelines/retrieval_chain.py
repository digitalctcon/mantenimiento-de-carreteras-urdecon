from langchain_astradb import AstraDBVectorStore
from langchain.chains import create_retrieval_chain
import os
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.utils import get_available_tasks

# Load environment variables from .env file
load_dotenv()

# VARIABLES, TOKENS AND KEYS
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")

# Initialize Retrieval Chain
def get_retrieval_chain(search_kwargs=None):
    """
    Create and return a retrieval chain using AstraDB and OpenAI embeddings.

    Args:
        search_kwargs (dict, optional): Search parameters for the retriever.

    Returns:
        RetrievalQA: The retrieval-based question-answering chain.
    """
    # Initialize embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = AstraDBVectorStore(
        collection_name="urdecontest1",
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAMESPACE,
    )
    # Set up retriever
    if search_kwargs is None:
        search_kwargs = {"k": 3}
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Initialize OpenAI model via ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_template("""Utiliza el siguiente contexto para responder a la pregunta del usuario.
    
    <context>
    {context}
    </context>
                                              
    Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

    Pregunta: {input}""")

    # Create a chain that passes documents to an LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retrieval chain that connects the retriever to the LLM
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

def extract_ubicacion(query):
    """
    Extract all 'ubicacion' from the query.

    Args:
        query (str): The natural language query.

    Returns:
        list: A list of extracted 'ubicacion' or an empty list if none found.
    """
    available_tasks = get_available_tasks()
    found_tasks = []
    for task in available_tasks:
        if task in query:
            found_tasks.append(task)
    return found_tasks

# Query the Retrieval Chain
def retrieve_report(query):
    """
    Query the retrieval chain and get results.

    Args:
        query (str): The natural language query to search the vectorstore.

    Returns:
        dict: The result containing the answer and source documents.
    """
    # Extract ubicacion from query
    ubicaciones = extract_ubicacion(query)
    #print(f"Ubicaciones: {ubicaciones}")
    # Set up search kwargs with filter if ubicaciones are found
    search_kwargs = {"k": 3}
    if ubicaciones:
        search_kwargs["filter"] = {"ubicacion": {"$in": ubicaciones}}
    #print(f"Search kwargs: {search_kwargs}")
    # Get retrieval chain with search kwargs
    retrieval_chain = get_retrieval_chain(search_kwargs)
    
    response = retrieval_chain.invoke({"input": query})
    return response
