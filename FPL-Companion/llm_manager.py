from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
from langchain_openai import ChatOpenAI
from types import SimpleNamespace
from typing import Optional, List, Any
from pydantic import Field
from config import settings
from fpl_rag_core import IntentClassifier, EntityExtractor
from cypher_queries import CypherQueryLibrary
from vector_search import VectorSearch
import json

class CustomHFWrapper(LLM):
    """
    Direct wrapper for HuggingFace InferenceClient to avoid LangChain version conflicts.
    Mimics the user's GemmaLangChainWrapper but for general use.
    """
    client: Any = Field(...)
    repo_id: str = "google/gemma-2-2b-it" # Switched to Gemma as per user example
    
    @property
    def _llm_type(self) -> str:
        return "custom_hf_client"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop:
            raise ValueError("stop kwargs are not permitted.")
        
        try:
             # Use chat_completion (User's working method)
             response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
             )
             return response.choices[0].message.content
        except Exception as e:
            return f"Error in HF Call: {repr(e)}"

class RAGManager:
    def __init__(self, llm_type="openai"):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.cypher_lib = CypherQueryLibrary()
        self.vector_search = VectorSearch()
        self.llm = self._setup_llm(llm_type)
        
    def _setup_llm(self, llm_type):
        if llm_type == "openai" and settings.OPENAI_API_KEY:
            return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-3.5-turbo")
        elif llm_type == "huggingface" and settings.HUGGINGFACE_API_KEY:
            # Use direct InferenceClient with our Custom Wrapper
            client = InferenceClient(token=settings.HUGGINGFACE_API_KEY, model="google/gemma-2-2b-it")
            return CustomHFWrapper(client=client, repo_id="google/gemma-2-2b-it")
        else:
            return None

    def close(self):
        # self.intent_classifier.close() # No close method needed
        self.entity_extractor.close()
        self.cypher_lib.close()
        self.vector_search.close()

    def process_query(self, user_query, retrieval_strategy="hybrid"):
        # 1. Classification & Extraction
        intent = self.intent_classifier.classify(user_query)
        entities = self.entity_extractor.extract_entities(user_query)
        
        context = []
        executed_queries = []
        
        # 2. Retrieval Strategy
        # Strategy A: Baseline (Cypher)
        if retrieval_strategy in ["baseline", "hybrid"]:
            if intent == "stats":
                # Check which template fits
                extracted_params = {}
                if entities['players']: extracted_params['player_name'] = entities['players'][0]
                if entities['seasons']: extracted_params['season'] = entities['seasons'][0]
                if entities['teams']: extracted_params['team_name'] = entities['teams'][0] # or opponent
                
                # Default season if missing
                if 'season' not in extracted_params: extracted_params['season'] = "2022-23"
                
                queries_to_run = []
                if 'player_name' in extracted_params:
                     queries_to_run.append("get_player_stats")
                elif 'team_name' in extracted_params:
                     queries_to_run.append("get_team_fixtures") # Example
                else: 
                     # Only run fallback Cypher if we are strictly in baseline mode
                     # Otherwise, let Vector Search handle this likely generic/semantic query
                     if retrieval_strategy == "baseline":
                        queries_to_run.append("get_total_gameweeks")
                
                for q_name in queries_to_run:
                    try:
                        data = self.cypher_lib.execute_query(q_name, extracted_params)
                        context.append(f"Cypher Result ({q_name}): {json.dumps(data)}")
                        executed_queries.append(q_name)
                    except Exception as e:
                        print(f"Error running {q_name}: {e}")

        # Strategy B: Vector Search
        if retrieval_strategy in ["embeddings", "hybrid"]:
            if intent in ["recommendation", "general", "comparison"] or not context:
                # Use vector search for semantic/recommendation queries
                results = self.vector_search.search_similar_players(user_query)
                context.append(f"Vector Search Findings: {json.dumps(results)}")
                executed_queries.append("vector_search")
        
        # 3. LLM Generation
        context_str = "\n".join(context)
        
        system_prompt = f"""You are an FPL (Fantasy Premier League) Expert Assistant.
        Context Information from Knowledge Graph:
        {context_str}
        
        Task: Answer the user's question using ONLY the context provided. If the answer is not in the context, say "I don't have enough information in my database."
        """
        
        if self.llm:
            try:
                # Simplifying for both Chat and minimal LLM interfaces
                if hasattr(self.llm, "invoke"):
                    messages = [
                        ("system", system_prompt),
                        ("human", user_query),
                    ]
                    response = self.llm.invoke(messages)
                    return response.content if hasattr(response, 'content') else str(response), context, executed_queries
                # Custom Wrapper Fallback (if invoke vs call confusion)
                elif hasattr(self.llm, "_call"):
                     # For our CustomHFWrapper (which inherits from LLM), invoke() should work via LangChain base class.
                     # But if not, we can construct the prompt manually.
                     full_prompt = f"{system_prompt}\nUser Question: {user_query}\nAnswer:"
                     response = self.llm(full_prompt) # LLM.__call__ calls _call
                     return response, context, executed_queries
                else:
                    return f"LLM Configured but invoke failed. Context: {context_str}", context, executed_queries
            except Exception as e:
                return f"Error calling LLM: {e}. Context: {context_str}", context, executed_queries
        else:
            return f"[No LLM API Key] Context Found: {context_str}", context, executed_queries
