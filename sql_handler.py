from sqlalchemy import create_engine, text
import json
from typing import Optional, Dict, Any
import os

class SQLQueryHandler:
    def __init__(self):
        self.engine = self._create_db_engine()
        
    def _create_db_engine(self):
        # Load database configuration from environment variables
        db_user = os.environ.get("DB_USER")
        db_password = os.environ.get("DB_PASSWORD")
        public_ip = os.environ.get("PUBLIC_IP")
        port = os.environ.get("DB_PORT")
        db_name = os.environ.get("DB_NAME")

        if not all([db_user, db_password, public_ip, port, db_name]):
            # Get secrets from streamlit secrets
            db_user = st.secrets["DB_USER"]
            db_password = st.secrets["DB_PASSWORD"]
            public_ip = st.secrets["PUBLIC_IP"]
            port = st.secrets["DB_PORT"]
            db_name = st.secrets["DB_NAME"]

        # if not all([db_user, db_password, public_ip, port, db_name]):
        #     raise ValueError("One or more required database environment variables are missing.")

        # Create connection string
        connection_url = f"mysql+pymysql://{db_user}:{db_password}@{public_ip}:{port}/{db_name}"
        return create_engine(connection_url)
    
    def execute_query(self, sql_query: str, params: Optional[Dict[str, Any]] = None, fetch: bool = True):
        """Execute SQL query with optional parameters and return results"""
        try:
            with self.engine.connect() as connection:
                with connection.begin():  # Start transaction
                    if params:
                        result = connection.execute(text(sql_query), params)
                    else:
                        result = connection.execute(text(sql_query))
                    
                    if fetch:
                        return result.fetchall()
                    return None
                    
        except Exception as e:
            print(f"Database Error: {e}")
            return None
        finally:
            self.engine.dispose() 