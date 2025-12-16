from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
# from langchain_openai import ChatOpenAI # Removed
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

class MetricsTracker:
    def __init__(self, cost_per_1m_input=0.20, cost_per_1m_output=0.60):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_input = cost_per_1m_input
        self.cost_output = cost_per_1m_output

    def count_tokens(self, text):
        # Approximate: 4 chars per token
        if not text: return 0
        return len(text) // 4

    def track_request(self, input_text, output_text):
        in_tok = self.count_tokens(input_text)
        out_tok = self.count_tokens(output_text)
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        return {
            "input_tokens": in_tok, 
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "cost": (in_tok * self.cost_input / 1_000_000) + (out_tok * self.cost_output / 1_000_000)
        }

class RAGManager:
    def __init__(self, llm_type="google/gemma-2-2b-it", embedding_model="all-MiniLM-L6-v2"):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.cypher_lib = CypherQueryLibrary()
        self.vector_search = VectorSearch(model_name=embedding_model)
        self.llm = self._setup_llm(llm_type)
        self.metrics = MetricsTracker() # Initialize tracker
        
    def _setup_llm(self, llm_type):
        # Only supporting Free HF Models now
        if llm_type in ["google/gemma-2-2b-it", "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]:
            if settings.HUGGINGFACE_API_KEY:
                # Use direct InferenceClient with our Custom Wrapper
                client = InferenceClient(token=settings.HUGGINGFACE_API_KEY, model=llm_type)
                return CustomHFWrapper(client=client, repo_id=llm_type)
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
        context = []
        executed_queries = []
        queries_to_run = []
        
        # 2. Retrieval Strategy
        # Strategy A: Baseline (Cypher)
        if retrieval_strategy in ["baseline", "hybrid"]:
            if intent == "stats":
                # Check which template fits
                extracted_params = {}
                # queries_to_run = [] # Moved to top
                if entities['players']: extracted_params['player_name'] = entities['players'][0]
                if entities['seasons']: extracted_params['season'] = entities['seasons'][0]
                if entities['teams']: extracted_params['team'] = entities['teams'][0] # Changed key to 'team' to match Cypher param
                
                # Check for Specific Stats Keywords
                if any(x in user_query.lower() for x in ["clean sheet", "cleansheet", "clean sheets", "cleansheets"]) and 'player_name' in extracted_params:
                     extracted_params['season'] = extracted_params.get('season', "2022-23")
                     queries_to_run.append("get_player_cleansheets")

                if any(x in user_query.lower() for x in ["ict", "influence", "creativity", "threat"]) and not 'player_name' in extracted_params:
                     queries_to_run.append("get_highest_ict_player")

                if "max" in user_query.lower() or "record" in user_query.lower() or "highest" in user_query.lower():
                     queries_to_run.append("get_max_points_gw")

                if "goal" in user_query.lower() and "more than" in user_query.lower():
                     # Simple regex for number? defaulting to 10 provided in query lib
                     queries_to_run.append("get_players_with_min_goals")
                
                # Default season if missing
                # Default season if missing
                # Default season if missing
                if 'season' not in extracted_params: extracted_params['season'] = "2022-23"
                
                if 'player_name' in extracted_params and 'team' in extracted_params:
                     # 1. Player vs Team
                     extracted_params['opponent'] = extracted_params['team']
                     queries_to_run.append("get_player_performance_vs_team")
                
                elif 'player_name' in extracted_params:
                     # 2. Player Stats (GW or Season)
                     if entities.get('gameweeks'):
                         extracted_params['gw'] = entities['gameweeks'][0]
                         queries_to_run.append("get_player_gw_stats")
                     else:
                         queries_to_run.append("get_player_stats")

                elif 'team' in extracted_params:
                     # 3. Team Fixtures
                     extracted_params['team_name'] = extracted_params['team']
                     queries_to_run.append("get_team_fixtures")
                
                else: 
                     # 4. Fallback
                     # Only run fallback Cypher if we are strictly in baseline mode
                     # Otherwise, let Vector Search handle this likely generic/semantic query
                     if retrieval_strategy == "baseline":
                        queries_to_run.append("get_total_gameweeks")

                # Enhancement: If Position is mentioned in Stats, get top players for that pos (and team if present)
                if entities['positions']:
                    extracted_params['position'] = entities['positions'][0]
                    queries_to_run.append("get_top_players_by_position")
                
                for q_name in queries_to_run:
                    try:
                        data = self.cypher_lib.execute_query(q_name, extracted_params)
                        context.append(f"Cypher Result ({q_name}): {json.dumps(data)}")
                        executed_queries.append(q_name)
                    except Exception as e:
                        print(f"Error running {q_name}: {e}")

                        print(f"Error running {q_name}: {e}")

        # Strategy B: Vector Search & Hybrid Recommendation
        if retrieval_strategy in ["embeddings", "hybrid"]:
            # If explicit intent is recommendation OR general fallback
            if intent == "comparison" and len(entities['players']) >= 2:
                 # 0. Precise Player Comparison (New)
                 p1 = entities['players'][0]
                 p2 = entities['players'][1]
                 try:
                     data = self.cypher_lib.execute_query("compare_players", {"season": "2022-23", "p1": p1, "p2": p2})
                     context.append(f"Comparison Data: {json.dumps(data)}")
                     executed_queries.append("compare_players")
                 except Exception as e:
                     print(f"Comparison failed: {e}")

            if intent in ["recommendation", "general", "comparison"] or (not context and retrieval_strategy != "baseline"):
                
                # 1. Vector Search (Semantic)
                filters = {}
                if entities['positions']: filters['position'] = entities['positions'][0]
                if entities['teams']: filters['team'] = entities['teams'][0]
                if entities['players']: filters['reference_player'] = entities['players'][0]
                
                # Logic: Only exclude the player if we are explicitly recommending NEW players.
                # If asking "Who is Salah" (general) or "Compare Salah" (comparison), we want to see Salah.
                if intent in ["recommendation"]:
                    filters['exclude_self'] = True
                else:
                    filters['exclude_self'] = False

                results = self.vector_search.search_similar_players(user_query, filters=filters)
                context.append(f"Vector Search Findings: {json.dumps(results)}")
                executed_queries.append("vector_search")
                
                # 2. Hybrid Augmentation: Add Statistical Top Performers for Recommendation OR General Team Analysis
                if intent == "recommendation" or (intent in ["general", "comparison"] and entities['teams']):
                    # Determine sort criteria or specialized query? For now, generic top performers.
                    # Pass entities to filter if possible? 
                    # Currently get_top_performing_players doesn't filter by team/pos in the Cypher text, 
                    # but we could call get_top_players_by_position if position is known.
                    
                    hybrid_q = "get_top_performing_players"
                    hybrid_params = {"season": "2022-23"} # Default
                    
                    if entities['positions']:
                        hybrid_q = "get_top_players_by_position"
                        hybrid_params["position"] = entities['positions'][0]
                    
                    if entities['teams']:
                        hybrid_params["team"] = entities['teams'][0]

                    # Note: We don't have a "get_top_players_by_team" generic query ready in cypher_queries for stats summation, 
                    # but "get_player_stats" is specific to one player.
                    # If team is known, we might want top players OF that team.
                    # Ideally, we'd add 'get_top_players_by_team' to cypher_queries.py too, but effectively 
                    # vector search with team filter might cover 'best X in team Y'.
                    # BUT vector search was failing on quantitative 'best'.
                    
                    # Let's run the statistical query to ground the LLM
                    try:
                        data = self.cypher_lib.execute_query(hybrid_q, hybrid_params)
                        context.append(f"Statistical Top Performers ({hybrid_q}): {json.dumps(data)}")
                        executed_queries.append(hybrid_q)
                    except Exception as e:
                        print(f"Hybrid Cypher failed: {e}")
        
        # 3. LLM Generation
        context_str = "\n".join(context)
        
        system_prompt = f"""You are an FPL (Fantasy Premier League) Expert Assistant.
        
        Information Retrieved from Knowledge Graph:
        {context_str}
        
        Instructions:
        1. If Vector Search Findings are provided, use them to identify relevant players.
        2. If Statistical Top Performers are provided, PRIORITIZE them for recommendations.
        3. consider the "avg_form" and "next_opponent" fields to assess difficulty and recent performance.
        4. If Cypher Results are provided, answer using those exact stats.
        5. DO NOT ask follow-up questions. Use the data provided to give a direct answer.
        6. If truly no relevant data exists, say "I don't have that information in my database."
        """
        
        response_text = ""
        full_input_text = f"{system_prompt}\nUser: {user_query}"
        
        if self.llm:
            try:
                # Simplifying for both Chat and minimal LLM interfaces
                if hasattr(self.llm, "invoke"):
                    messages = [
                        ("system", system_prompt),
                        ("human", user_query),
                    ]
                    response = self.llm.invoke(messages)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                # Custom Wrapper Fallback (if invoke vs call confusion)
                elif hasattr(self.llm, "_call"):
                     # For our CustomHFWrapper (which inherits from LLM), invoke() should work via LangChain base class.
                     # But if not, we can construct the prompt manually.
                     full_prompt = f"{system_prompt}\nUser Question: {user_query}\nAnswer:"
                     response = self.llm(full_prompt) # LLM.__call__ calls _call
                     response_text = response
                     full_input_text = full_prompt
                else:
                    response_text = f"LLM Configured but invoke failed. Context: {context_str}"
            except Exception as e:
                response_text = f"Error calling LLM: {e}. Context: {context_str}"
        else:
            response_text = f"[No LLM API Key] Context Found: {context_str}"
            
        # Track Metrics
        usage_stats = self.metrics.track_request(full_input_text, response_text)
            
        return response_text, context, executed_queries, usage_stats
