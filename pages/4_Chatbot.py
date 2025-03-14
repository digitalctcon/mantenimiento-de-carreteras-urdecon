__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pipelines.chatbot_chain import create_chatbot_workflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the chatbot workflow
chatbot_app = create_chatbot_workflow()

st.subheader("Busca información sobre los proyectos en marcha")

# Session state for conversation history
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "chat_thread_1"

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for msg in st.session_state["chat_history"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Chat input
if user_input := st.chat_input("Escribe aquí tu consulta..."):
    # Append user input to history
    st.session_state["chat_history"].append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Invoke the chatbot workflow
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    result = chatbot_app.invoke({"messages": st.session_state["chat_history"]}, config)

    # Add assistant's response to chat history
    ai_response = result["messages"][-1]
    st.session_state["chat_history"].append(ai_response)
    st.chat_message("assistant").write(ai_response.content)
