#__import__('pysqlite3')
import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import requests
import datetime
from huggingface_hub import InferenceClient
from langchain_pipelines.generate_report_chain import generate_report
import os 
from src.utils import get_channel_id, get_available_tasks, get_tasks_by_project, get_project_description
from src.utils import store_report_in_chroma
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# VARIABLES, TOKENS AND KEYS
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fetch tasks specific to Proyecto 2
project_name = "Proyecto 1"
project_description = get_project_description(project_name)
task_options = get_tasks_by_project(project_name)

client = OpenAI()

# Functions to interact with OpenAI API
def query_whisper(audio_value):
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_value,
        response_format="text"
    )
    return transcription

def send_to_slack(channel, message):
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    payload = {"channel": channel, "text": message}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Streamlit Interface
st.subheader(f"{project_description}")

nombre_persona = st.text_input("Responsable:")
tarea = st.selectbox("Selecciona la tarea:", task_options)
fecha = st.date_input("Fecha:", datetime.date.today(), format="DD/MM/YYYY")
audio_value = st.audio_input("Grabar audio:")

if audio_value:
    st.audio(audio_value)

    # Step 1: Transcription with Whisper
    transcription_response = query_whisper(audio_value)
    print(transcription_response)
    if transcription_response:
        st.subheader("Transcripción del audio:")
        st.write(transcription_response)
        
        # Metadata
        metadata = {
            "nombre_persona": nombre_persona,
            "ubicacion": tarea,
            "fecha": fecha.strftime("%d/%m/%Y"),
        }

        # Generate report using LangChain's pipeline
        st.write("Generando informe...")
        informe = generate_report(metadata, transcription_response)
        st.write(informe)
        # Store the initial report in session state
        if "latest_report" not in st.session_state:
            st.session_state.latest_report = informe

        # User feedback
        st.subheader("Modificar informe")
        feedback = st.text_input("Indica si hay algo que quieras modificar del informe:")
        if st.button("Modificar informe"):
            if feedback:
                # Integrate feedback as an additional input to the model
                metadata_with_feedback = {
                    **metadata,
                    "feedback": feedback,
                }

                # Update the transcription with the feedback context
                feedback_prompt = f"""
                Informe original:
                {st.session_state.latest_report}

                Modificaciones solicitadas por el usuario:
                {feedback}

                Genera un informe actualizado basado en las modificaciones proporcionadas.
                """

                # Generate refined report
                refined_informe = generate_report(metadata, feedback_prompt)
                informe = refined_informe
                st.session_state.latest_report = refined_informe
                st.subheader("Informe modificado:")
                st.write(refined_informe)

    # Slack Integration
    slack_channel = get_channel_id(tarea)
    informe_slack = informe.replace("**", "*").replace("- ", "• ")
    if st.button("Enviar informe"):
        store_report_in_chroma(informe, metadata)
        slack_response = send_to_slack(slack_channel, informe_slack)
        if slack_response.get("ok"):
            st.success("Informe enviado correctamente.")
        else:
            st.error(f"Error al enviar informe: {slack_response.get('error')}")
