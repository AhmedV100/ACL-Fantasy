from neo4j import GraphDatabase
from config import settings
import sys

print(f"Testing connection to: {settings.NEO4J_URI}")
print(f"User: {settings.NEO4J_USERNAME}")
# Mask password for security in logs
print(f"Password length: {len(settings.NEO4J_PASSWORD)}")

try:
    driver = GraphDatabase.driver(
        settings.NEO4J_URI, 
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("✅ Connection Verified Successfully!")
    driver.close()
except Exception as e:
    import traceback
    traceback.print_exc()
    # print(f"❌ Connection Failed: {e}")
    sys.exit(1)
