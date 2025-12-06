import re
from config import settings
from neo4j import GraphDatabase

class IntentClassifier:
    def __init__(self):
        self.intents = {
            "stats": ["how many", "stats", "points", "goals", "assists", "clean sheets", "red cards", "yellow cards", "minutes", "score"],
            "recommendation": ["recommend", "suggest", "best", "top", "pick", "transfer", "buy", "sell"],
            "comparison": ["compare", "better", "vs", "versus", "difference"],
            "general": ["what is", "explain", "tell me", "who is"]
        }

    def classify(self, query):
        query = query.lower()
        scores = {intent: 0 for intent in self.intents}
        
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query:
                    scores[intent] += 1
        
        # Default to stats if fuzzy, or max score
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return "general"
        return best_intent

class EntityExtractor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        self.knowledge_cache = self._build_cache()

    def close(self):
        self.driver.close()

    def _build_cache(self):
        """Fetch all known players, teams, and positions from DB to aid extraction"""
        cache = {
            "players": {},
            "teams": {},
            "positions": {},
            "seasons": {}
        }
        try:
            with self.driver.session() as session:
                # Players
                result = session.run("MATCH (p:Player) RETURN p.player_name")
                cache["players"] = {row["p.player_name"].lower(): row["p.player_name"] for row in result}
                
                # Teams
                result = session.run("MATCH (t:Team) RETURN t.name")
                cache["teams"] = {row["t.name"].lower(): row["t.name"] for row in result}
                
                # Positions
                result = session.run("MATCH (p:Position) RETURN p.name")
                cache["positions"] = {row["p.name"].lower(): row["p.name"] for row in result}
                
                # Seasons
                result = session.run("MATCH (s:Season) RETURN s.season_name")
                cache["seasons"] = {row["s.season_name"]: row["s.season_name"] for row in result}
                
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j for cache building. {e}")
        return cache

    def extract_entities(self, query):
        query_lower = query.lower()
        entities = {
            "players": [],
            "teams": [],
            "positions": [],
            "seasons": []
        }
        
        # Exact/Fuzzy Match with Cache
        for name_lower, original_name in self.knowledge_cache["players"].items():
            # 1. Full name match
            if name_lower in query_lower:
                entities["players"].append(original_name)
                continue
            
            # 2. Last name / Partial match (simple heuristic)
            # If the player name has multiple parts, check if the last part (surname) is in query
            parts = name_lower.split()
            if len(parts) > 1:
                last_name = parts[-1]
                # Use word boundary to avoid "Son" matching "Season"
                if re.search(r'\b' + re.escape(last_name) + r'\b', query_lower):
                    entities["players"].append(original_name)
                    
        for team_lower, original_name in self.knowledge_cache["teams"].items():
            if team_lower in query_lower:
                entities["teams"].append(original_name)
                
        for pos_lower, original_name in self.knowledge_cache["positions"].items():
            if re.search(r'\b' + re.escape(pos_lower) + r'\b', query_lower):
                entities["positions"].append(original_name)
        
        # Seasons (regex for YYYY-YY)
        season_matches = re.findall(r'\d{4}-\d{2}', query)
        for season in season_matches:
            if season in self.knowledge_cache["seasons"]:
                entities["seasons"].append(season)
            elif "20"+season.split("-")[1] in str(self.knowledge_cache["seasons"]): # simplified
                 entities["seasons"].append(season)
        
        # Default season if none found ? Maybe not here.
        
        return entities

if __name__ == "__main__":
    # Test
    classifier = IntentClassifier()
    print(f"Intent: {classifier.classify('How many goals did Haaland score?')}")
    
    extractor = EntityExtractor()
    print(f"Entities: {extractor.extract_entities('How many goals did Haaland score for Man City in 2022-23?')}")
    extractor.close()
