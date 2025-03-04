__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append(r"D:\02_Construccion Digital\01 En ejecucion\Cheque PLN Urdecon\App using langchain\urdecon-pln-mantenimiento-de-carreteras")

import streamlit as st
from src.utils import get_channel_id, get_available_tasks, get_tasks_by_project, get_project_description
from langchain_pipelines.retrieval_chain import retrieve_report

st.subheader("Busca información sobre los proyectos en marcha")

# User input for query
query = st.text_input("Escribe tu consulta:")
DEBUG = False

if st.button("Enviar"):
    if query:
        # Retrieve reports using the chain
        result = retrieve_report(query)

        # Display the answer
        st.write("### Respuesta:")
        st.write(result.get("result"))

        # Display source documents
        if(DEBUG):
            st.write("### Documentos relacionados:")
            source_docs = result.get("source_documents", [])
            if source_docs:
                for doc in source_docs:
                    st.write(f"**Informe:** {doc.page_content}")
                    st.write(f"**Metadatos:** {doc.metadata}")
                    st.write("---")
            else:
                st.write("No se encontraron documentos relacionados.")
    else:
        st.warning("Por favor, introduce una consulta.")
