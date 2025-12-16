import csv
from neo4j import GraphDatabase
import os
import time

class FPLKnowledgeGraphBatch:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all existing nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def create_constraints(self):
        """Create uniqueness constraints for all node types"""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT season_name IF NOT EXISTS FOR (s:Season) REQUIRE s.season_name IS UNIQUE")
            session.run("CREATE CONSTRAINT gameweek_id IF NOT EXISTS FOR (g:Gameweek) REQUIRE (g.season, g.GW_number) IS UNIQUE")
            session.run("CREATE CONSTRAINT fixture_id IF NOT EXISTS FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS UNIQUE")
            session.run("CREATE CONSTRAINT team_name IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT player_id IF NOT EXISTS FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS UNIQUE")
            session.run("CREATE CONSTRAINT position_name IF NOT EXISTS FOR (pos:Position) REQUIRE pos.name IS UNIQUE")
            print("Constraints created")

    def load_csv_data(self, csv_file):
        """Load and return CSV data"""
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def batch_data(self, data, batch_size=1000):
        """Yield successive chunks of data."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def create_seasons_batch(self, data):
        """Create Season nodes in batch"""
        seasons = list(set(row['season'] for row in data))
        
        def create_seasons_tx(tx, batch):
            query = """
            UNWIND $batch AS season_name
            MERGE (s:Season {season_name: season_name})
            """
            tx.run(query, batch=batch)

        with self.driver.session() as session:
            session.execute_write(create_seasons_tx, seasons)
        print(f"Created {len(seasons)} Season nodes")

    def create_gameweeks_batch(self, data):
        """Create Gameweek nodes and HAS_GW relationships in batch"""
        gameweeks = {}
        for row in data:
            key = (row['season'], row['GW'])
            gameweeks[key] = {'season': row['season'], 'gw_number': int(row['GW'])}
        unique_gws = list(gameweeks.values())
        
        def create_gw_tx(tx, batch):
            query = """
            UNWIND $batch AS row
            MERGE (s:Season {season_name: row.season})
            MERGE (g:Gameweek {season: row.season, GW_number: row.gw_number})
            MERGE (s)-[:HAS_GW]->(g)
            """
            tx.run(query, batch=batch)
        
        with self.driver.session() as session:
            for batch in self.batch_data(unique_gws):
                session.execute_write(create_gw_tx, batch)
        print(f"Created/Merged {len(unique_gws)} Gameweek nodes")

    def create_teams_batch(self, data):
        """Create Team nodes in batch"""
        teams = set()
        for row in data:
            teams.add(row['home_team'])
            teams.add(row['away_team'])
            
        def create_teams_tx(tx, batch):
            query = """
            UNWIND $batch AS name
            MERGE (t:Team {name: name})
            """
            tx.run(query, batch=batch)
            
        with self.driver.session() as session:
            session.execute_write(create_teams_tx, list(teams))
        print(f"Created {len(teams)} Team nodes")
    
    def create_positions_batch(self, data):
        """Create Position nodes in batch"""
        positions = set(row['position'] for row in data)
        
        def create_positions_tx(tx, batch):
            query = """
            UNWIND $batch AS name
            MERGE (p:Position {name: name})
            """
            tx.run(query, batch=batch)
            
        with self.driver.session() as session:
            session.execute_write(create_positions_tx, list(positions))
        print(f"Created {len(positions)} Position nodes")

    def create_fixtures_batch(self, data):
        """Create Fixture nodes and relationships in batch"""
        fixtures = {}
        for row in data:
            key = (row['season'], row['fixture'])
            if key not in fixtures:
                fixtures[key] = {
                    'season': row['season'],
                    'fixture_number': int(row['fixture']),
                    'kickoff_time': row['kickoff_time'],
                    'gw_number': int(row['GW']),
                    'home_team': row['home_team'],
                    'away_team': row['away_team']
                }
        unique_fixtures = list(fixtures.values())
        
        def create_fixtures_tx(tx, batch):
            query = """
            UNWIND $batch AS row
            MATCH (g:Gameweek {season: row.season, GW_number: row.gw_number})
            MATCH (home:Team {name: row.home_team})
            MATCH (away:Team {name: row.away_team})
            MERGE (f:Fixture {season: row.season, fixture_number: row.fixture_number})
            ON CREATE SET f.kickoff_time = row.kickoff_time
            MERGE (g)-[:HAS_FIXTURE]->(f)
            MERGE (f)-[:HAS_HOME_TEAM]->(home)
            MERGE (f)-[:HAS_AWAY_TEAM]->(away)
            """
            tx.run(query, batch=batch)

        print(f"Processing {len(unique_fixtures)} unique fixtures...")
        with self.driver.session() as session:
            for batch in self.batch_data(unique_fixtures):
                session.execute_write(create_fixtures_tx, batch)
        print("Fixtures created")

    def create_players_batch(self, data):
        """Create Player nodes, Positions, and Team links in batch"""
        players = {}
        # 1. Pre-process to identify positions and all teams seen
        for row in data:
            key = (row['name'], row['element'])
            if key not in players:
                players[key] = {
                    'name': row['name'],
                    'element': int(row['element']),
                    'positions': set(),
                    'teams_seen': [] # Track teams to infer primary team
                }
            players[key]['positions'].add(row['position'])
            players[key]['teams_seen'].extend([row['home_team'], row['away_team']])
        
        # 2. Prepare final list with primary team inferred
        player_list = []
        from collections import Counter
        
        for p in players.values():
            p['positions'] = list(p['positions'])
            
            # Infer Team
            team_counts = Counter(p['teams_seen'])
            p['primary_team'] = team_counts.most_common(1)[0][0] if team_counts else None
            
            # Remove helper list to save bandwidth
            del p['teams_seen']
            
            player_list.append(p)
            
        def create_players_tx(tx, batch):
            query = """
            UNWIND $batch AS row
            MERGE (p:Player {player_name: row.name, player_element: row.element})
            
            // Positions
            FOREACH (pos_name IN row.positions | 
                MERGE (pos:Position {name: pos_name})
                MERGE (p)-[:PLAYS_AS]->(pos)
            )
            
            // Team Link
            FOREACH (t_name IN CASE WHEN row.primary_team IS NOT NULL THEN [row.primary_team] ELSE [] END |
                MERGE (t:Team {name: t_name})
                MERGE (p)-[:PLAYS_FOR]->(t)
            )
            """
            tx.run(query, batch=batch)

        print(f"Processing {len(player_list)} unique players...")
        with self.driver.session() as session:
            for batch in self.batch_data(player_list):
                session.execute_write(create_players_tx, batch)
        print("Players, Position links, and Team links created")

    def create_performances_batch(self, data):
        """Create PLAYED_IN relationships with stats in batch"""
        
        # 1. Build Player -> Team Map for was_home calculation
        player_teams = {}
        temp_teams = {} # (name, element) -> list of teams
        for row in data:
             key = (row['name'], row['element'])
             if key not in temp_teams: temp_teams[key] = []
             temp_teams[key].append(row['home_team'])
             temp_teams[key].append(row['away_team'])
        
        from collections import Counter
        for k, v in temp_teams.items():
            if v:
                player_teams[k] = Counter(v).most_common(1)[0][0]
        
        processed_rows = []
        for row in data:
            key = (row['name'], row['element'])
            p_team = player_teams.get(key)
            was_home = (row['home_team'] == p_team) if p_team else False

            processed_rows.append({
                'player_name': row['name'],
                'player_element': int(row['element']),
                'season': row['season'],
                'fixture_number': int(row['fixture']),
                'minutes': int(row['minutes']),
                'goals_scored': int(row['goals_scored']),
                'assists': int(row['assists']),
                'total_points': int(row['total_points']),
                'bonus': int(row['bonus']),
                'clean_sheets': int(row['clean_sheets']),
                'goals_conceded': int(row['goals_conceded']),
                'own_goals': int(row['own_goals']),
                'penalties_saved': int(row['penalties_saved']),
                'penalties_missed': int(row['penalties_missed']),
                'yellow_cards': int(row['yellow_cards']),
                'red_cards': int(row['red_cards']),
                'saves': int(row['saves']),
                'bps': float(row['bps']),
                'influence': float(row['influence']),
                'creativity': float(row['creativity']),
                'threat': float(row['threat']),
                'ict_index': float(row['ict_index']),
                'form': float(row['form']),
                'value': int(row['value']),
                'was_home': was_home
            })
            
        def create_perfs_tx(tx, batch):
            query = """
            UNWIND $batch AS row
            MATCH (p:Player {player_name: row.player_name, player_element: row.player_element})
            MATCH (f:Fixture {season: row.season, fixture_number: row.fixture_number})
            MERGE (p)-[r:PLAYED_IN]->(f)
            SET r.minutes = row.minutes,
                r.goals_scored = row.goals_scored,
                r.assists = row.assists,
                r.total_points = row.total_points,
                r.bonus = row.bonus,
                r.clean_sheets = row.clean_sheets,
                r.goals_conceded = row.goals_conceded,
                r.own_goals = row.own_goals,
                r.penalties_saved = row.penalties_saved,
                r.penalties_missed = row.penalties_missed,
                r.yellow_cards = row.yellow_cards,
                r.red_cards = row.red_cards,
                r.saves = row.saves,
                r.bps = row.bps,
                r.influence = row.influence,
                r.creativity = row.creativity,
                r.threat = row.threat,
                r.ict_index = row.ict_index,
                r.ict_index = row.ict_index,
                r.form = row.form,
                r.value = row.value,
                r.was_home = row.was_home
            """
            tx.run(query, batch=batch)

        print(f"Processing {len(processed_rows)} performance records...")
        
        start_time = time.time()
        with self.driver.session() as session:
            total = len(processed_rows)
            # Reduced batch size to 1000 for stability
            for i, batch in enumerate(self.batch_data(processed_rows, batch_size=1000)):
                session.execute_write(create_perfs_tx, batch)
                if i % 10 == 0:
                    print(f"  Progress: {(i+1)*1000}/{total} records...")
        
        end_time = time.time()
        print(f"Performances created in {end_time - start_time:.2f} seconds")

    def build_knowledge_graph(self, csv_file):
        print("Starting BATCH Knowledge Graph construction...")
        data = self.load_csv_data(csv_file)
        print(f"Loaded {len(data)} rows")
        
        self.clear_database()
        self.create_constraints()
        
        self.create_seasons_batch(data)
        self.create_gameweeks_batch(data)
        self.create_teams_batch(data)
        self.create_positions_batch(data)
        self.create_fixtures_batch(data)
        self.create_players_batch(data)
        self.create_performances_batch(data)
        
        print("\nBATCH Construction Completed Successfully!")

def main():
    from config import settings
    
    uri = settings.NEO4J_URI
    username = settings.NEO4J_USERNAME
    password = settings.NEO4J_PASSWORD
    csv_file = 'fpl_two_seasons.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return

    kg = FPLKnowledgeGraphBatch(uri, username, password)
    try:
        kg.build_knowledge_graph(csv_file)
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        kg.close()

if __name__ == "__main__":
    main()
