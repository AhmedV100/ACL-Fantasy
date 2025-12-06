import streamlit as st
from llm_manager import RAGManager
from vector_search import VectorSearch
import os

# Page Config
st.set_page_config(page_title="FPL Journey Companion", page_icon="⚽", layout="wide")

st.title("⚽ FPL Graph-RAG Assistant")
st.markdown("Your AI-powered assistant for Fantasy Premier League insights, grounded in a Neo4j Knowledge Graph.")

# Sidebar
st.sidebar.header("Configuration")
retrieval_mode = st.sidebar.selectbox("Retrieval Strategy", ["hybrid", "baseline", "embeddings"])
llm_model = st.sidebar.selectbox("LLM Model", ["openai", "huggingface", "none"])

if st.sidebar.button("Rebuild Embeddings"):
    with st.spinner("Generating Embeddings... this may take a while"):
        try:
            vs = VectorSearch()
            vs.create_embeddings()
            vs.close()
            st.sidebar.success("Embeddings created!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    # We initialize RAG manager lightly, or re-init if config changes could be better
    # For now, simplistic re-init per run or persistent? 
    # Better to re-init to capture sidebar changes instantly for prototype
    pass 

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("Retrieved Context & Queries"):
                st.code("\n".join(message["context"]), language="json")
                st.write("Executed Queries:", message["queries"])

# Chat Input
if prompt := st.chat_input("Ask about players, stats, or recommendations..."):
    # Render User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process
    try:
        rag = RAGManager(llm_type=llm_model)
        with st.spinner("Thinking..."):
            response_text, context, queries = rag.process_query(prompt, retrieval_strategy=retrieval_mode)
            rag.close()
        
        # Render Assistant Message
        with st.chat_message("assistant"):
            st.markdown(response_text)
            with st.expander("Retrieved Context & Queries"):
                st.code("\n".join(context), language="json")
                st.write("Executed Queries:", queries)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "context": context,
            "queries": queries
        })
    except Exception as e:
        st.error(f"Error: {e}")
