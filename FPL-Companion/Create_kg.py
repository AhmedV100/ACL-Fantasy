import csv
from neo4j import GraphDatabase
import os

class FPLKnowledgeGraph:
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
            # Season constraint
            session.run("""
                CREATE CONSTRAINT season_name IF NOT EXISTS
                FOR (s:Season) REQUIRE s.season_name IS UNIQUE
            """)
            
            # Gameweek constraint (composite: season + GW_number)
            session.run("""
                CREATE CONSTRAINT gameweek_id IF NOT EXISTS
                FOR (g:Gameweek) REQUIRE (g.season, g.GW_number) IS UNIQUE
            """)
            
            # Fixture constraint (composite: season + fixture_number)
            session.run("""
                CREATE CONSTRAINT fixture_id IF NOT EXISTS
                FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS UNIQUE
            """)
            
            # Team constraint
            session.run("""
                CREATE CONSTRAINT team_name IF NOT EXISTS
                FOR (t:Team) REQUIRE t.name IS UNIQUE
            """)
            
            # Player constraint (composite: player_name + player_element)
            session.run("""
                CREATE CONSTRAINT player_id IF NOT EXISTS
                FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS UNIQUE
            """)
            
            # Position constraint
            session.run("""
                CREATE CONSTRAINT position_name IF NOT EXISTS
                FOR (pos:Position) REQUIRE pos.name IS UNIQUE
            """)
            
            print("Constraints created")
    
    def load_csv_data(self, csv_file):
        """Load and return CSV data"""
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def create_seasons(self, data):
        """Create Season nodes"""
        seasons = set(row['season'] for row in data)
        
        with self.driver.session() as session:
            for season in seasons:
                session.run("""
                    MERGE (s:Season {season_name: $season_name})
                """, season_name=season)
        
        print(f"Created {len(seasons)} Season nodes")
    
    def create_gameweeks(self, data):
        """Create Gameweek nodes and HAS_GW relationships"""
        gameweeks = set((row['season'], row['GW']) for row in data)
        
        with self.driver.session() as session:
            for season, gw in gameweeks:
                session.run("""
                    MERGE (g:Gameweek {season: $season, GW_number: $gw_number})
                """, season=season, gw_number=int(gw))
                
                # Create HAS_GW relationship
                session.run("""
                    MATCH (s:Season {season_name: $season})
                    MATCH (g:Gameweek {season: $season, GW_number: $gw_number})
                    MERGE (s)-[:HAS_GW]->(g)
                """, season=season, gw_number=int(gw))
        
        print(f"Created {len(gameweeks)} Gameweek nodes with HAS_GW relationships")
    
    def create_teams(self, data):
        """Create Team nodes"""
        teams = set()
        for row in data:
            teams.add(row['home_team'])
            teams.add(row['away_team'])
        
        with self.driver.session() as session:
            for team in teams:
                session.run("""
                    MERGE (t:Team {name: $name})
                """, name=team)
        
        print(f"Created {len(teams)} Team nodes")
    
    def create_fixtures(self, data):
        """Create Fixture nodes and relationships"""
        fixtures = {}
        for row in data:
            key = (row['season'], row['fixture'])
            if key not in fixtures:
                fixtures[key] = {
                    'season': row['season'],
                    'fixture_number': row['fixture'],
                    'kickoff_time': row['kickoff_time'],
                    'GW': row['GW'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team']
                }
        
        with self.driver.session() as session:
            for fixture_data in fixtures.values():
                # Create Fixture node
                session.run("""
                    MERGE (f:Fixture {
                        season: $season, 
                        fixture_number: $fixture_number
                    })
                    SET f.kickoff_time = $kickoff_time
                """, 
                    season=fixture_data['season'],
                    fixture_number=int(fixture_data['fixture_number']),
                    kickoff_time=fixture_data['kickoff_time']
                )
                
                # Create HAS_FIXTURE relationship
                session.run("""
                    MATCH (g:Gameweek {season: $season, GW_number: $gw_number})
                    MATCH (f:Fixture {season: $season, fixture_number: $fixture_number})
                    MERGE (g)-[:HAS_FIXTURE]->(f)
                """, 
                    season=fixture_data['season'],
                    gw_number=int(fixture_data['GW']),
                    fixture_number=int(fixture_data['fixture_number'])
                )
                
                # Create HAS_HOME_TEAM relationship
                session.run("""
                    MATCH (f:Fixture {season: $season, fixture_number: $fixture_number})
                    MATCH (t:Team {name: $team_name})
                    MERGE (f)-[:HAS_HOME_TEAM]->(t)
                """, 
                    season=fixture_data['season'],
                    fixture_number=int(fixture_data['fixture_number']),
                    team_name=fixture_data['home_team']
                )
                
                # Create HAS_AWAY_TEAM relationship
                session.run("""
                    MATCH (f:Fixture {season: $season, fixture_number: $fixture_number})
                    MATCH (t:Team {name: $team_name})
                    MERGE (f)-[:HAS_AWAY_TEAM]->(t)
                """, 
                    season=fixture_data['season'],
                    fixture_number=int(fixture_data['fixture_number']),
                    team_name=fixture_data['away_team']
                )
        
        print(f"Created {len(fixtures)} Fixture nodes with relationships")
    
    def create_positions(self, data):
        """Create Position nodes"""
        positions = set(row['position'] for row in data)
        
        with self.driver.session() as session:
            for position in positions:
                session.run("""
                    MERGE (pos:Position {name: $name})
                """, name=position)
        
        print(f"Created {len(positions)} Position nodes")
    
    def create_players_and_relationships(self, data):
        """Create Player nodes and all their relationships"""
        players = {}
        
        # Group data by player
        for row in data:
            key = (row['name'], row['element'])
            if key not in players:
                players[key] = {
                    'name': row['name'],
                    'element': row['element'],
                    'positions': set(),
                    'fixtures': [],
                    'teams_seen': [] # Track all teams seen in fixtures to infer actual team
                }
            players[key]['positions'].add(row['position'])
            players[key]['fixtures'].append(row)
            players[key]['teams_seen'].extend([row['home_team'], row['away_team']])
        
        with self.driver.session() as session:
            for player_data in players.values():
                # --- Infer Team Logic ---
                # A player's team must be present in their fixtures.
                # The team that appears most frequently in (home_team, away_team) across all fixtures is the player's team.
                # (Handling transfers: Simply taking the most frequent for now, assuming one main team per season)
                from collections import Counter
                team_counts = Counter(player_data['teams_seen'])
                if team_counts:
                    primary_team_name = team_counts.most_common(1)[0][0]
                else:
                    primary_team_name = None

                # Create Player node
                session.run("""
                    MERGE (p:Player {
                        player_name: $player_name, 
                        player_element: $player_element
                    })
                """, 
                    player_name=player_data['name'],
                    player_element=int(player_data['element'])
                )
                
                # Create PLAYS_AS relationships
                for position in player_data['positions']:
                    session.run("""
                        MATCH (p:Player {player_name: $player_name, player_element: $player_element})
                        MATCH (pos:Position {name: $position})
                        MERGE (p)-[:PLAYS_AS]->(pos)
                    """, 
                        player_name=player_data['name'],
                        player_element=int(player_data['element']),
                        position=position
                    )
                
                 # Create PLAYS_FOR relationship (New)
                if primary_team_name:
                    session.run("""
                        MATCH (p:Player {player_name: $player_name, player_element: $player_element})
                        MATCH (t:Team {name: $team_name})
                        MERGE (p)-[:PLAYS_FOR]->(t)
                    """,
                        player_name=player_data['name'],
                        player_element=int(player_data['element']),
                        team_name=primary_team_name
                    )

                # Create PLAYED_IN relationships with properties
                for fixture in player_data['fixtures']:
                    # Calculate was_home
                    was_home = (fixture['home_team'] == primary_team_name)

                    session.run("""
                        MATCH (p:Player {player_name: $player_name, player_element: $player_element})
                        MATCH (f:Fixture {season: $season, fixture_number: $fixture_number})
                        MERGE (p)-[r:PLAYED_IN]->(f)
                        SET r.minutes = $minutes,
                            r.goals_scored = $goals_scored,
                            r.assists = $assists,
                            r.total_points = $total_points,
                            r.bonus = $bonus,
                            r.clean_sheets = $clean_sheets,
                            r.goals_conceded = $goals_conceded,
                            r.own_goals = $own_goals,
                            r.penalties_saved = $penalties_saved,
                            r.penalties_missed = $penalties_missed,
                            r.yellow_cards = $yellow_cards,
                            r.red_cards = $red_cards,
                            r.saves = $saves,
                            r.bps = $bps,
                            r.influence = $influence,
                            r.creativity = $creativity,
                            r.threat = $threat,
                            r.ict_index = $ict_index,
                            r.form = $form,
                            r.was_home = $was_home
                    """, 
                        player_name=player_data['name'],
                        player_element=int(player_data['element']),
                        season=fixture['season'],
                        fixture_number=int(fixture['fixture']),
                        minutes=int(fixture['minutes']),
                        goals_scored=int(fixture['goals_scored']),
                        assists=int(fixture['assists']),
                        total_points=int(fixture['total_points']),
                        bonus=int(fixture['bonus']),
                        clean_sheets=int(fixture['clean_sheets']),
                        goals_conceded=int(fixture['goals_conceded']),
                        own_goals=int(fixture['own_goals']),
                        penalties_saved=int(fixture['penalties_saved']),
                        penalties_missed=int(fixture['penalties_missed']),
                        yellow_cards=int(fixture['yellow_cards']),
                        red_cards=int(fixture['red_cards']),
                        saves=int(fixture['saves']),
                        bps=float(fixture['bps']),
                        influence=float(fixture['influence']),
                        creativity=float(fixture['creativity']),
                        threat=float(fixture['threat']),
                        ict_index=float(fixture['ict_index']),
                        form=float(fixture['form']),
                        was_home=was_home
                    )
        
        print(f"Created {len(players)} Player nodes with relationships (PLAYS_FOR, PLAYED_IN)")
    
    def build_knowledge_graph(self, csv_file):
        """Main method to build the entire knowledge graph"""
        print("Starting Knowledge Graph construction...")
        
        # Load data
        print(f"Loading data from {csv_file}...")
        data = self.load_csv_data(csv_file)
        print(f"Loaded {len(data)} rows")
        
        # Clear existing data
        self.clear_database()
        
        # Create constraints
        self.create_constraints()
        
        # Build graph step by step
        self.create_seasons(data)
        self.create_gameweeks(data)
        self.create_teams(data)
        self.create_fixtures(data)
        self.create_positions(data)
        self.create_players_and_relationships(data)
        
        print("\nKnowledge Graph construction completed!")


def read_config(config_file='config.txt'):
    """Read Neo4j configuration from config.txt"""
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def main():
    # Read configuration from shared settings
    from config import settings
    
    uri = settings.NEO4J_URI
    username = settings.NEO4J_USERNAME
    password = settings.NEO4J_PASSWORD
    
    # CSV file name
    csv_file = 'fpl_two_seasons.csv'
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found in current directory")
        return
    
    # Create knowledge graph
    kg = FPLKnowledgeGraph(uri, username, password)
    
    try:
        kg.build_knowledge_graph(csv_file)
    except Exception as e:
        print(f"Error during knowledge graph creation: {e}")
        raise
    finally:
        kg.close()


if __name__ == "__main__":
    main()