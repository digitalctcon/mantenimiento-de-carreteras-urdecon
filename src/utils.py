import sqlite3
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Function to establish a database connection
def get_connection():
    """Establish a connection to the SQLite database."""
    try:
        conn = sqlite3.connect("data/db/mantenimiento_de_carreteras.db")  # Adjust path if needed
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

def get_available_tasks():
    """Fetch the list of available tasks (task names) from the database."""
    conn = get_connection()
    if not conn:
        return []
    try:
        query = "SELECT name FROM tareas"
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return [row[0] for row in results]  # Return a list of task names
    except sqlite3.Error as e:
        print(f"Error fetching tasks: {e}")
        return []
    finally:
        conn.close()


# Function to retrieve the channel_id based on task_name
def get_channel_id(task_name):
    """Retrieve the Slack channel_id based on the task_name."""
    conn = get_connection()
    if not conn:
        return None
    try:
        query = """
            SELECT channel_id
            FROM canales_slack
            WHERE channel_name = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (task_name.lower(),))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        print(f"Database query error: {e}")
        return None
    finally:
        conn.close()

def get_tasks_by_project(project_name):
    """Fetch tasks assigned to a specific project."""
    conn = get_connection()
    if not conn:
        return []
    try:
        query = """
            SELECT t.name 
            FROM tareas t
            JOIN proyectos p ON t.project_id = p.id
            WHERE p.name = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (project_name,))
        results = cursor.fetchall()
        return [row[0] for row in results]  # Return task names
    except sqlite3.Error as e:
        print(f"Error fetching tasks for project '{project_name}': {e}")
        return []
    finally:
        conn.close()

def get_project_description(project_name):
    """Fetch the description of a project based on its name."""
    conn = get_connection()
    if not conn:
        return "Descripción no disponible."
    try:
        query = """
            SELECT description 
            FROM proyectos 
            WHERE name = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (project_name,))
        result = cursor.fetchone()
        return result[0] if result else "Descripción no disponible."
    except sqlite3.Error as e:
        print(f"Error fetching project description: {e}")
        return "Descripción no disponible."
    finally:
        conn.close()

def setup_chroma():
    """Initialize Chroma vector store with HuggingFace embeddings."""
    # Using HuggingFace model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize Chroma vector store
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vectorstore

def store_report_in_chroma(report_text, metadata):
    """
    Store a report in the Chroma vector store.

    Args:
        report_text (str): The main content of the report.
        metadata (dict): Metadata such as date, responsible person, and task.
    """
    vectorstore = setup_chroma()

    # Create a document with report content and metadata
    document = Document(
        page_content=report_text,  # The actual report content
        metadata=metadata          # Metadata like date, person, task
    )

    # Add the document to the vector store
    vectorstore.add_documents([document])
    
    print("Report stored successfully!")

