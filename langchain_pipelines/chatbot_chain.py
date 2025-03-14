from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_pipelines.retrieval_chain import retriever
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize the LLM
def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
    )

# Define the chatbot memory workflow
def create_chatbot_workflow():
    llm = initialize_llm()
    workflow = StateGraph(state_schema=MessagesState)

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retriever])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Step 2: Execute the retrieval.
    tools = ToolNode([retriever])

    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "Eres un asistente para tareas de respuesta a preguntas."
            "Usa los siguientes fragmentos de contexto para responder la pregunta."
            "Si no sabes la respuesta, di que no la sabes."
            "Usa un máximo de cuatro frases y mantén la respuesta concisa"
            "\n\n"
            f"{docs_content}"
        )

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}

    # Define the node and edge in the workflow
    workflow.add_edge(START, "query_or_respond")
    workflow.add_node("query_or_respond", query_or_respond)
    workflow.add_node("tools", tools)
    workflow.add_node("generate", generate)
    workflow.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    workflow.add_edge("tools", "generate")
    workflow.add_edge("generate", END)

    # Add memory persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app