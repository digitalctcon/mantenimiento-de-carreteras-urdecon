__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append(r"D:\02_Construccion Digital\01 En ejecucion\Cheque PLN Urdecon\App using langchain\urdecon-pln-mantenimiento-de-carreteras")

import streamlit as st
import requests
import datetime
from huggingface_hub import InferenceClient
from langchain_pipelines.generate_report_chain import generate_report
import os 
from src.utils import get_channel_id, get_available_tasks, get_tasks_by_project, get_project_description
from src.utils import store_report_in_chroma

# VARIABLES, TOKENS AND KEYS
HF_TOKEN = st.secrets["HF_TOKEN"]
SLACK_BOT_TOKEN = st.secrets["SLACK_BOT_TOKEN"]
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

WHISPER_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Fetch tasks specific to Proyecto 2
project_name = "Proyecto 2"
project_description = get_project_description(project_name)
task_options = get_tasks_by_project(project_name)

# Functions to interact with Hugging Face APIs
def query_whisper(audio_bytes):
    response = requests.post(WHISPER_API_URL, headers=WHISPER_HEADERS, data=audio_bytes)
    return response.json()

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
    transcription_response = query_whisper(audio_value.read())
    if "text" in transcription_response:
        transcription = transcription_response["text"]
        st.subheader("Transcripción del audio:")
        st.write(transcription)
        #transcription = "Hola, soy Luis, estamos en la carretera que conecta el pueblo con la ciudad, justo en el kilómetro 12. Hemos estado reparando una zona donde el asfalto estaba bastante desgastado y tenía varias grietas profundas. Ya hemos terminado de rellenar dos grietas grandes y aplicado una capa de sellado, pero aún nos queda reparar un tramo de unos 50 metros. El tráfico está siendo desviado temporalmente por un carril alterno, y aunque hay algo de congestión, no hemos tenido incidentes hasta ahora. Por otro lado, notamos que el drenaje en este tramo parece estar obstruido, ya que hay acumulación de agua a un lado de la vía. Quizás sería bueno programar una revisión más detallada para evitar problemas en la temporada de lluvias. Voy a seguir aquí hasta que terminemos la reparación de este tramo. En unas horas te actualizo sobre el avance."
        
        # Metadata
        metadata = {
            "nombre_persona": nombre_persona,
            "ubicacion": tarea,
            "fecha": fecha.strftime("%d/%m/%Y"),
        }

        # Generate report using LangChain's pipeline
        st.write("Generando informe...")
        informe = generate_report(metadata, transcription)
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
