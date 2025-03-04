import streamlit as st

st.header("Sistema inteligente para la gestión del mantenimiento de carreteras")

st.markdown("""
Esta aplicación está diseñada para **gestionar proyectos de mantenimiento de carreteras** de manera eficiente.
Consta de varias funcionalidades que facilitan la generación de informes y búsqueda de datos históricos. 

### Funcionalidades principales
1. **Proyectos activos:**
    - En las páginas de proyectos, puedes **grabar audios** para registrar observaciones en campo.
    - El sistema genera **informes detallados** basados en tus descripciones, con opción de modificación.
    - Se almacena toda la información en una base de datos para consultas futuras.
        
        
2. **Chatbot de búsqueda:**
    - Usa el buscador para realizar **consultas inteligentes** sobre los informes generados previamente.
    - Por ejemplo, puedes hacer preguntas como: _"¿Qué problemas hubo en la RM-16?"_
        
""")