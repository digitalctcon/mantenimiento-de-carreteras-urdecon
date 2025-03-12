from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
import streamlit as st

# VARIABLES, TOKENS AND KEYS
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

report_structure = """
- Fecha
- Responsable
- Ubicación
- Descripción detallada del trabajo realizado
"""

# Step 1: Define the System and Human Prompts
report_system_template_str = """
Eres un experto en mantenimiento de carreteras. Tu trabajo es generar informes claros y detallados basándote en la información proporcionada por el usuario.
El informe debe tener la siguiente estructura:

{report_structure}

Por favor, no inventes información y utiliza solo los datos proporcionados por el usuario. Si no hay ninguna información para añadir a Descripción detallada del trabajo realizado, escribe "Sin información relevante".
El formato de tu respuesta debe ser simplemente el informe generado, no añadas más explicaciones. 
"""

report_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["report_structure"],
        template=report_system_template_str,
    )
)

report_human_template_str = """
Contexto proporcionado sobre la tarea realizada:

Fecha: {fecha}
Responsable: {nombre_persona}
Ubicación: {ubicacion}
Información detallada: {transcription}

Esta Información detallada viene de un audio, por lo que puede haber errores en la transcripción. Tenlo en cuenta a la hora de generar el informe y arregla lo que sea necesario.
"""

report_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["fecha", "nombre_persona", "ubicacion", "transcription"],
        template=report_human_template_str,
    )
)

# Step 2: Combine Prompts into a ChatPromptTemplate
report_prompt_template = ChatPromptTemplate(
    input_variables=["report_structure","fecha", "nombre_persona", "ubicacion", "transcription"],
    messages=[report_system_prompt, report_human_prompt],
)

def initialize_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
    )
    return llm

chat_model = initialize_llm()

# Step 4: Combine into a Chain
report_chain = (
    report_prompt_template
    | chat_model
    | StrOutputParser()
)

# Step 5: Use the Chain
def generate_report(metadata, transcription):
    """
    Generate a report using metadata and transcription or feedback context.
    If the transcription contains feedback, the report will be refined based on the feedback.
    """
    response = report_chain.invoke({
        "report_structure": report_structure,
        "fecha": metadata.get("fecha", "No especificada"),
        "nombre_persona": metadata.get("nombre_persona", "Persona de mantenimiento"),
        "ubicacion": metadata.get("ubicacion", "No especificada"),
        "transcription": transcription,  # This can be the initial or feedback-enriched transcription
    })
    return response