from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import settings

class VectorSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        self.model_name = model_name
        # Map models to specific index names and dimensions to keep them separate
        if "mpnet" in model_name:
             self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
             self.index_name = "player_stats_index_mpnet"
             self.property_name = "embedding_mpnet"
             self.dim = 768
        else:
             self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
             self.index_name = "player_stats_index_minilm"
             self.property_name = "embedding_minilm"
             self.dim = 384

    def close(self):
        self.driver.close()

    def create_embeddings(self, season="2022-23"):
        """
        Create embeddings using the selected model.
        """
        print(f"Fetching player data for embedding using {self.model_name}...")
        query = """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, pos, sum(r.total_points) as points, sum(r.goals_scored) as goals, sum(r.assists) as assists
        RETURN p.player_name as name, pos.name as position, points, goals, assists, elementId(p) as id
        """
        
        with self.driver.session() as session:
            # 1. Fetch Data
            results = session.run(query, season=season).data()
            
            # 2. Create Texts
            texts = []
            for r in results:
                text = f"Player: {r['name']}, Position: {r['position']}, Points: {r['points']}, Goals: {r['goals']}, Assists: {r['assists']}"
                texts.append(text)
            
            # 3. Generate Embeddings
            print(f"Generating embeddings with {self.model_name}...")
            embeddings = self.model.encode(texts)
            
            # 4. Store in Neo4j
            print(f"Storing embeddings in Neo4j property: {self.property_name}...")
            for i, r in enumerate(results):
                # We use dynamic property setting via formatting or apoc, but safe param usage is key.
                # Since property_name is internal/trusted, f-string is acceptable here for the property key.
                cypher = f"""
                MATCH (p:Player) WHERE elementId(p) = $id
                CALL db.create.setNodeVectorProperty(p, '{self.property_name}', $embedding)
                """
                session.run(cypher, id=r['id'], embedding=embeddings[i].tolist())
            
            # 5. Create Index (if not exists)
            index_cypher = f"""
            CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
            FOR (p:Player)
            ON (p.{self.property_name})
            OPTIONS {{indexConfig: {{
             `vector.dimensions`: {self.dim},
             `vector.similarity_function`: 'cosine'
            }}}}
            """
            session.run(index_cypher)
            print(f"Embeddings created and index '{self.index_name}' built.")

    def search_similar_players(self, query_text, limit=3, filters=None):
        """
        Search for players similar to the query text (e.g., 'high scoring forward')
        """
        if filters is None:
            filters = {}

        query_embedding = self.model.encode(query_text).tolist()
        
        # Try to detect if this is a "similar to X" query to enable quality filtering
        reference_player_name = filters.get('reference_player')
        min_quality_points = 20  # Default minimum
        
        if reference_player_name:
            # Fetch reference player's stats to set quality threshold
            with self.driver.session() as session:
                ref_query = """
                MATCH (p:Player {player_name: $name})-[r:PLAYED_IN]->(f:Fixture {season: '2022-23'})
                RETURN sum(r.total_points) as total_points
                """
                result = session.run(ref_query, name=reference_player_name).single()
                if result and result['total_points']:
                    # Require similar players to have at least 50% of reference player's points
                    min_quality_points = int(result['total_points'] * 0.5)
        
        position_clause = ""
        param_map = {'limit': limit * 10, 'embedding': query_embedding, 'min_points': min_quality_points}
        
        if filters.get('position'):
            position_clause = f"""
            MATCH (node)-[:PLAYS_AS]->(pos:Position)
            WHERE pos.name CONTAINS $position_filter
            """
            param_map['position_filter'] = filters['position']

        team_clause = ""
        if filters.get('team'):
            team_clause = f"""
            MATCH (node)-[:PLAYS_FOR]->(t:Team)
            WHERE t.name CONTAINS $team_filter
            """
            param_map['team_filter'] = filters['team']

        # Exclude reference player if present
        exclude_clause = ""
        if filters.get('reference_player'):
            exclude_clause = "AND node.player_name <> $ref_player_name"
            param_map['ref_player_name'] = filters['reference_player']

        cypher = f"""
        CALL db.index.vector.queryNodes('{self.index_name}', $limit, $embedding)
        YIELD node, score
        {position_clause}
        {team_clause}
        // Fetch stats without embedding
        MATCH (node)-[:PLAYS_AS]->(pos:Position)
        WHERE 1=1 {exclude_clause}
        OPTIONAL MATCH (node)-[played:PLAYED_IN]->(f:Fixture {{season: '2022-23'}})
        WITH node, score, pos, 
             sum(played.total_points) as total_points, 
             sum(played.goals_scored) as goals, 
             sum(played.assists) as assists
        WHERE total_points >= $min_points
        RETURN {{
          player_name: node.player_name, 
          position: pos.name,
          team_name: head([(node)-[:PLAYS_FOR]->(t) | t.name]),
          total_points: total_points,
          goals: goals,
          assists: assists
        }} as player_data, score
        ORDER BY score DESC
        LIMIT 5
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, **param_map)
                return [dict(record) for record in result]
            except Exception as e:
                print(f"Vector search failed (Index missing?): {e}")
                return []

if __name__ == "__main__":
    # Test
    vs = VectorSearch()
    # vs.create_embeddings() # Uncomment to run once
    # print(vs.search_similar_players("Top striker with many goals"))
    vs.close()
