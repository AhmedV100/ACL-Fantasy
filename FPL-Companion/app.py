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

# LLM Selection (Meeting requirement for 3 models)
llm_model = st.sidebar.selectbox("LLM Model", [
    "google/gemma-2-2b-it", 
    "meta-llama/Llama-3.1-8B-Instruct", 
    "mistralai/Mistral-7B-Instruct-v0.2"
])

# Embedding Experiment
st.sidebar.markdown("---")
st.sidebar.subheader("Experiment: Embeddings")
embedding_model = st.sidebar.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])

if st.sidebar.button("Rebuild Embeddings"):
    with st.spinner(f"Generating Embeddings using {embedding_model}..."):
        try:
            vs = VectorSearch(model_name=embedding_model)
            vs.create_embeddings()
            vs.close()
            st.sidebar.success(f"Embeddings created for {embedding_model}!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("Retrieved Context & Queries"):
                st.code("\n".join(message["context"]), language="json")
                st.write("Executed Queries:", message["queries"])
        if "metrics" in message:
             st.caption(f"⏱️ Response Time: {message['metrics']['time']:.2f}s | Model: {message['metrics']['model']}")

import time # Import locally if needed, or ensure top-level import

# Chat Input
if prompt := st.chat_input("Ask about players, stats, or recommendations..."):
    # Render User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process
    try:
        rag = RAGManager(llm_type=llm_model, embedding_model=embedding_model)
        
        start_time = time.time()
        with st.spinner("Thinking..."):
            response_text, context, queries, usage_stats = rag.process_query(prompt, retrieval_strategy=retrieval_mode)
            rag.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Render Assistant Message
        with st.chat_message("assistant"):
            st.markdown(response_text)
            
            # Quantitative Metrics Display
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.caption(f"⏱️ Time: {elapsed_time:.2f}s")
            with metrics_col2:
                st.caption(f"🪙 Tokens: {usage_stats['total_tokens']} ({usage_stats['input_tokens']} in / {usage_stats['output_tokens']} out)")
            with metrics_col3:
                st.caption(f"💰 Est. Cost: ${usage_stats['cost']:.5f}")
            
            # Context Expander
            with st.expander("Retrieved Context & Queries"):
                st.code("\n".join(context), language="json")
                st.write("Executed Queries:", queries)

            # --- Graph Visualization Snippet ---
            if context:
                with st.expander("🕸️ Graph Visualization"):
                    try:
                        import graphviz
                        import json
                        import re
                        
                        graph = graphviz.Digraph()
                        graph.attr(rankdir='LR', size='8,5')
                        graph.attr('node', shape='oval', style='filled', color='lightblue')
                        
                        nodes_added = set()
                        edges_added = set()

                        def nice_label(label):
                            return re.sub(r'[^a-zA-Z0-9 ]', '', str(label))[:15]

                        for ctx_item in context:
                            # Try to extract JSON part
                            match = re.search(r'(\{.*\}|\[.*\])', ctx_item, re.DOTALL)
                            if match:
                                try:
                                    data = json.loads(match.group(1))
                                    if isinstance(data, list):
                                        items = data
                                    else:
                                        items = [data]
                                    
                                    for item in items:
                                        if isinstance(item, dict):
                                            # Identify Player
                                            p_name = item.get('player_name') or item.get('name') or item.get('player')
                                            if p_name:
                                                p_id = f"P_{nice_label(p_name)}"
                                                if p_id not in nodes_added:
                                                    graph.node(p_id, label=str(p_name), color='#90ee90') # Green for Player
                                                    nodes_added.add(p_id)
                                            
                                            # Identify Team / Opponent
                                            t_name = item.get('team_name') or item.get('team') or item.get('opponent')
                                            if t_name:
                                                t_id = f"T_{nice_label(t_name)}"
                                                if t_id not in nodes_added:
                                                    graph.node(t_id, label=str(t_name), color='#add8e6') # Blue for Team
                                                    nodes_added.add(t_id)
                                                
                                                # Edge Logic
                                                if p_name:
                                                    edge_key = f"{p_id}-{t_id}"
                                                    if edge_key not in edges_added:
                                                        graph.edge(p_id, t_id, label="RELATED")
                                                        edges_added.add(edge_key)

                                            # Identify Season (Contextual)
                                            season = item.get('season')
                                            if season and p_name:
                                                 s_id = f"S_{nice_label(season)}"
                                                 if s_id not in nodes_added:
                                                     graph.node(s_id, label=str(season), shape='box', color='#ffcccb')
                                                     nodes_added.add(s_id)
                                                 graph.edge(f"P_{nice_label(p_name)}", s_id, label="IN")

                                except Exception as json_err:
                                    pass # Skip malformed json

                        if nodes_added:
                            st.graphviz_chart(graph)
                        else:
                            st.info("No visualizeable entities found in context.")
                            
                    except ImportError:
                        st.warning("Graphviz library not found. Install it to see visualizations.")
                    except Exception as e:
                        st.error(f"Visualization Error: {e}")
            # -----------------------------------
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "context": context,
            "queries": queries,
            "metrics": {"time": elapsed_time, "model": llm_model}
        })
    except Exception as e:
        st.error(f"Error: {e}")
