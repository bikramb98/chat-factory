from sqlalchemy import create_engine, text
import json

class SQLQueryHandler:
    def __init__(self, config_path="db_config.json"):
        self.engine = self._create_db_engine(config_path)
        
    def _create_db_engine(self, config_path):
        # Load database configuration
        with open(config_path, "r") as file:
            config = json.load(file)
            
        # Create connection string
        connection_url = f"mysql+pymysql://{config['DB_USER']}:{config['DB_PASSWORD']}@{config['PUBLIC_IP']}:{config['PORT']}/{config['DB_NAME']}"
        return create_engine(connection_url)
    
    def execute_query(self, sql_query):
        """Execute SQL query and return results"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query))
                return result.fetchall()
        except Exception as e:
            print(f"Database Error: {e}")
            return None
        finally:
            self.engine.dispose() 