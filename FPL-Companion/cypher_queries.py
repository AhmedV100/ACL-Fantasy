from neo4j import GraphDatabase
from config import settings

class CypherQueryLibrary:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def execute_query(self, query_name, params=None):
        query_func = getattr(self, query_name, None)
        if query_func:
            cypher, clean_params = query_func(params)
            with self.driver.session() as session:
                result = session.run(cypher, **clean_params)
                return [dict(record) for record in result]
        else:
            raise ValueError(f"Query {query_name} not found")

    # 1. Player Stats by Season
    def get_player_stats(self, params):
        query = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture {season: $season})
        RETURN p.player_name, f.season, sum(r.total_points) as total_points, sum(r.goals_scored) as goals, sum(r.assists) as assists
        """
        return query, {"player_name": params.get("player_name"), "season": params.get("season")}

    # 1b. Player Stats by Gameweek (New)
    def get_player_gw_stats(self, params):
        query = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture {season: $season})
        MATCH (g:Gameweek {season: $season, GW_number: $gw})
        MATCH (g)-[:HAS_FIXTURE]->(f)
        RETURN p.player_name, f.season, g.GW_number, r.total_points, r.goals_scored, r.assists
        """
        return query, {
            "player_name": params.get("player_name"), 
            "season": params.get("season"),
            "gw": int(params.get("gw"))
        }

    # 2. Top Players by Position
    def get_top_players_by_position(self, params):
        query = """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        
        // Optional Team Filter
        MATCH (p)-[:PLAYS_FOR]->(t:Team)
        WHERE ($team IS NULL OR t.name = $team)

        RETURN p.player_name, pos.name, sum(r.total_points) as total_points
        ORDER BY total_points DESC
        LIMIT $limit
        """
        return query, {
            "position": params.get("position"), 
            "season": params.get("season"),
            "limit": int(params.get("limit", 5)),
            "team": params.get("team") # Pass None if missing
        }

    # 3. Team Fixtures
    def get_team_fixtures(self, params):
        query = """
        MATCH (t:Team {name: $team_name})<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f:Fixture {season: $season})
        RETURN f.fixture_number, f.kickoff_time, f.season
        ORDER BY f.fixture_number ASC
        LIMIT 5
        """
        return query, {"team_name": params.get("team_name"), "season": params.get("season")}

    # 4. Player Performance vs Team
    def get_player_performance_vs_team(self, params):
        query = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture {season: $season})
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {name: $opponent})
        WHERE NOT (t.name = $player_team)
        RETURN p.player_name, t.name as opponent, r.total_points, r.goals_scored
        """
        # Note: getting player_team is tricky without extra info, simplifying for now
        query_simple = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {name: $opponent})
        RETURN p.player_name, f.season, r.total_points, r.goals_scored
        """
        return query_simple, {"player_name": params.get("player_name"), "opponent": params.get("opponent")}

    # 5. Most/Least Expensive Players (using Form/ICT as proxy since cost isn't explicitly in create_kg logic above, checking r.value? No.)
    # Using ICT Index instead
    def get_highest_ict_player(self, params):
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        RETURN p.player_name, avg(r.ict_index) as avg_ict
        ORDER BY avg_ict DESC
        LIMIT 5
        """
        return query, {"season": params.get("season")}

    # 6. Player Clean Sheets
    def get_player_cleansheets(self, params):
        query = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture {season: $season})
        RETURN p.player_name, sum(r.clean_sheets) as clean_sheets
        """
        return query, {"player_name": params.get("player_name"), "season": params.get("season")}
    
    # 7. Total Gameweeks (from user queries.txt)
    def get_total_gameweeks(self, params):
        query = """
        MATCH (g:Gameweek)
        RETURN COUNT(g) AS total_gameweeks
        """
        return query, {}

    # 8. Max Points in a Gameweek (from user queries.txt)
    def get_max_points_gw(self, params):
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        RETURN MAX(r.total_points) AS max_points, p.player_name AS player
        ORDER BY max_points DESC
        LIMIT 1
        """
        return query, {"season": params.get("season")}

    # 9. Players with > X Goals
    def get_players_with_min_goals(self, params):
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, sum(r.goals_scored) as total_goals
        WHERE total_goals > $min_goals
        RETURN p.player_name, total_goals
        ORDER BY total_goals DESC
        """
        return query, {"season": params.get("season"), "min_goals": int(params.get("min_goals", 10))}

    # 10. Compare Two Players
    def compare_players(self, params):
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE p.player_name IN [$p1, $p2]
        RETURN p.player_name, sum(r.total_points) as points, sum(r.goals_scored) as goals, sum(r.assists) as assists
        """
        return query, {"season": params.get("season"), "p1": params.get("p1"), "p2": params.get("p2")}

    # 11. Top Performing Players (Generic)
        return query_improved, {
            "season": params.get("season", "2022-23"), 
            "limit": int(params.get("limit", 5)),
            "next_gw": next_gw,
            "team": params.get("team")
        } 

    # 11. Top Performing Players (Weighted by Form & Next Fixture)
    def get_top_performing_players(self, params):
        # Default to GW 37 to simulate "Next is 38" for historical data
        current_gw = int(params.get("current_gw", 37))
        next_gw = current_gw + 1
        
        query_improved = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        
        // Mandatory Team Filter if provided
        MATCH (p)-[:PLAYS_FOR]->(t:Team)
        WHERE ($team IS NULL OR t.name = $team)
        
        WITH p, t, sum(r.total_points) as total_points, avg(r.form) as avg_form, sum(r.goals_scored) as goals, sum(r.assists) as assists
        
        // Get Next Opponent for specific Next GW (Optional)
        OPTIONAL MATCH (t)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(next_f:Fixture {season: $season})<-[:HAS_FIXTURE]-(g:Gameweek {GW_number: $next_gw})
        OPTIONAL MATCH (next_f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(opp:Team)
        WHERE opp <> t
        
        RETURN p.player_name, total_points, avg_form, goals, assists, 
               coalesce(opp.name, 'None') as next_opponent
        ORDER BY (total_points + (avg_form * 8)) DESC
        LIMIT $limit
        """
        return query_improved, {
            "season": params.get("season", "2022-23"), 
            "limit": int(params.get("limit", 5)),
            "next_gw": next_gw,
            "team": params.get("team")
        }

    # 12. Player Recent Form (Last 5 Games)
    def get_player_recent_form(self, params):
        query = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture {season: $season})
        MATCH (g:Gameweek)-[:HAS_FIXTURE]->(f)
        RETURN p.player_name, f.season, g.GW_number, r.total_points, r.goals_scored, r.assists, r.bonus, r.was_home
        ORDER BY g.GW_number DESC
        LIMIT 5
        """
        return query, {"player_name": params.get("player_name"), "season": params.get("season")}

    # 13. Bonus Point Magnets
    def get_bonus_point_magnets(self, params):
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        RETURN p.player_name, sum(r.bonus) as total_bonus, sum(r.total_points) as total_points
        ORDER BY total_bonus DESC
        LIMIT 10
        """
        return query, {"season": params.get("season")}

    # 12. Player Home vs Away Performance (New)
    def get_player_home_away_performance(self, params):
        query = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture {season: $season})
        RETURN p.player_name, r.was_home as played_at_home, 
               avg(r.total_points) as avg_points, 
               sum(r.goals_scored) as total_goals, 
               sum(r.assists) as total_assists,
               count(r) as matches_played
        ORDER BY played_at_home DESC
        """
        return query, {"player_name": params.get("player_name"), "season": params.get("season")}

    # 13. Top Scorers by Team (New)
    def get_team_top_scorers(self, params):
        query = """
        MATCH (t:Team {name: $team_name})<-[:PLAYS_FOR]-(p:Player)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, sum(r.goals_scored) as goals, sum(r.assists) as assists, sum(r.total_points) as points
        WHERE goals > 0
        RETURN p.player_name, goals, assists, points
        ORDER BY goals DESC
        LIMIT 5
        """
        return query, {"team_name": params.get("team_name"), "season": params.get("season")}

    # 14. Value for Money Analysis (New)
    def get_player_value_analysis(self, params):
        query = """
        MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture {season: $season})
        RETURN p.player_name, 
               avg(r.value) / 10.0 as avg_price, 
               sum(r.total_points) as total_points,
               (toFloat(sum(r.total_points)) / (avg(r.value) / 10.0)) as points_per_million
        """
        return query, {"player_name": params.get("player_name"), "season": params.get("season")}

    # 15. Budget Filter (New)
    def get_players_by_budget(self, params):
        # Value in DB is int (e.g. 80 for 8.0)
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(:Fixture {season: $season})
        // Filter by Position if provided
        WHERE ($position IS NULL OR (p)-[:PLAYS_AS]->(:Position {name: $position}))
        WITH p, avg(r.value) as avg_val, sum(r.total_points) as total_points
        WHERE avg_val <= ($max_price * 10)
        RETURN p.player_name, avg_val/10.0 as cost, total_points
        ORDER BY total_points DESC
        LIMIT 10
        """
        return query, {
            "season": params.get("season", "2022-23"),
            "max_price": float(params.get("budget")),
            "position": params.get("position")
        }

