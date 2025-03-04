import streamlit as st

st.set_page_config(page_title="Mantenimiento de carreteras", layout="wide")

st.logo(
    "data/images/urdecon-img1.png",
    icon_image="data/images/urdecon-img1.png",
    size="large",
    link="https://urdecon.es/"
)

inicio = st.Page("pages/1_inicio.py", title="Cómo funciona la App", icon=":material/info:")
proyecto1 = st.Page("pages/2_RM-16,_RM-17,_RM-2,_RM-23,_RM-3.py", title="RM-16, RM-17, RM-2, RM-23, RM-3", icon=":material/lab_profile:")
proyecto2 = st.Page("pages/3_RM-602,_RM-608,_RM-609.py", title="RM-602, RM-608, RM-609", icon=":material/lab_profile:")
chatbot = st.Page("pages/4_Chatbot.py", title="Consulta de informes", icon=":material/search:")

pg = st.navigation(
    {
        "Acerca de": [inicio],
        "Proyectos": [proyecto1, proyecto2],
        "ChatBOT": [chatbot],
    }
)

pg.run()