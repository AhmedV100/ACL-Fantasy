from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import settings

class VectorSearch:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        self.index_name = "player_embeddings"

    def close(self):
        self.driver.close()

    def create_embeddings(self, season="2022-23"):
        """
        Create embeddings for players based on their aggregated stats in a season.
        Feature Vector: "Player Name, Position, Total Points: X, Goals: Y, Assists: Z"
        """
        print("Fetching player data for embedding...")
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
            print("Generating embeddings...")
            embeddings = self.model.encode(texts)
            
            # 4. Store in Neo4j
            print("Storing embeddings in Neo4j...")
            # Using Neo4j 5.x vector index syntax or setting properties directly
            # First, set the property
            for i, r in enumerate(results):
                session.run("""
                MATCH (p:Player) WHERE elementId(p) = $id
                CALL db.create.setNodeVectorProperty(p, 'embedding', $embedding)
                """, id=r['id'], embedding=embeddings[i].tolist())
            
            # 5. Create Index (if not exists)
            session.run("""
            CREATE VECTOR INDEX player_stats_index IF NOT EXISTS
            FOR (p:Player)
            ON (p.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 384,
             `vector.similarity_function`: 'cosine'
            }}
            """)
            print("Embeddings created and index built.")

    def search_similar_players(self, query_text, limit=3):
        """
        Search for players similar to the query text (e.g., 'high scoring forward')
        """
        query_embedding = self.model.encode(query_text).tolist()
        
        cypher = """
        CALL db.index.vector.queryNodes('player_stats_index', $limit, $embedding)
        YIELD node, score
        RETURN node.player_name, score
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, limit=limit, embedding=query_embedding)
            return [dict(record) for record in result]

if __name__ == "__main__":
    # Test
    vs = VectorSearch()
    # vs.create_embeddings() # Uncomment to run once
    # print(vs.search_similar_players("Top striker with many goals"))
    vs.close()
