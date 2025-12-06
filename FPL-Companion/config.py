import os

class Config:
    def __init__(self, config_file='config.txt'):
        self.config = self._read_config(config_file)
        
        # Neo4j Settings
        self.NEO4J_URI = os.getenv('NEO4J_URI', self.config.get('URI', 'neo4j://localhost:7687'))
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', self.config.get('USERNAME', 'neo4j'))
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', self.config.get('PASSWORD', 'password'))
        
        # LLM Settings
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', self.config.get('OPENAI_API_KEY'))
        self.HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', self.config.get('HUGGINGFACE_API_KEY'))
        
    def _read_config(self, config_file):
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        return config

# Global instance
settings = Config()
