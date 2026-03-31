# File: chatbot.py
# Description: StreamLit-based chatbot interface for LangGraph RAG system using ChromaDB and Qwen2.5 model

import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.config import load_agent_config
from rag_agent.ragtype import Message
from rag_agent.rag import load_graph


CONFIG = load_agent_config()
THREAD_ID = CONFIG.get("ui", {}).get("thread_id", "myrag")
RETRIEVAL_OPTIONS = ["sparse", "static", "dense"]
DEFAULT_RETRIEVAL = "static"


# st.title("Chat with your bot")

# select retrieval strategy
# Persist strategy in URL so browser refresh keeps the same selection.
query_strategy = st.query_params.get("retrieval", DEFAULT_RETRIEVAL)
if query_strategy not in RETRIEVAL_OPTIONS:
    query_strategy = DEFAULT_RETRIEVAL

retrieval_strategy = st.selectbox(
    "Retrieval strategy",
    options=RETRIEVAL_OPTIONS,
    index=RETRIEVAL_OPTIONS.index(query_strategy),
    help=(
        "Choose how documents are retrieved:\n"
        "- sparse: BM25 keyword search (no vector embeddings)\n"
        "- static: lightweight static embedding model (fast, lower quality)\n"
        "- dense: BGE dense embedding model (best semantic matching, slower)"
    ),
)

if st.query_params.get("retrieval") != retrieval_strategy:
    st.query_params["retrieval"] = retrieval_strategy

# only load the graph once
if "langgraph_app" not in st.session_state or st.session_state.get("active_retrieval_strategy") != retrieval_strategy:
    st.session_state.active_retrieval_strategy = retrieval_strategy
    st.session_state.messages = [] # clear the chat history
    st.session_state.langgraph_app = load_graph(st.session_state.active_retrieval_strategy) # reload the graph with defined retrieval strategy

app = st.session_state.langgraph_app

# Initialize chat history and config
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": THREAD_ID}}


# record message in frontend side
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask the bot..."):
    # Add user message
    st.session_state.messages.append(Message(role="user", content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # Run LangGraph with streaming
    assistant_container = st.chat_message("assistant")
    with assistant_container:
        with st.spinner("Thinking..."):

            # Stream the graph
            intermediate_outputs = []  # Collect intermediate steps
            answer = None
            for chunk in app.stream({"question": prompt}, st.session_state.config):
                for node, update in chunk.items():
                    if node == "search_documents":
                        docs = update["retrieved_docs"]
                    
                    elif node == "chain_of_thought":
                        cot = update["thought"]
                    
                    elif node == "generate_answer":
                        answer = update["answer"]

            with st.expander("Documents Retrieved"):
                st.text('\n\n'.join([doc for doc in docs]))

            # Display intermediate steps in an expander for cleanliness
            with st.expander("Chain of Thought"):
                st.text(cot)

        # Display final response
        if answer:
            st.write(answer)
            st.session_state.messages.append(Message(role="assistant", content=answer))
        else:
            st.write("No final response generated.")