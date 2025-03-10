from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json

@dataclass
class QueryLog:
    timestamp: datetime
    user_query: str
    query_type: str
    llm_responses: Dict[str, Any]
    sources: Optional[List[Dict[str, Any]]] = None
    feedback: Optional[str] = None
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'user_query': self.user_query,
            'query_type': self.query_type,
            'llm_responses': json.dumps(self.llm_responses),
            'sources': json.dumps(self.sources) if self.sources else None,
            'feedback': self.feedback
        }

class LogHandler:
    def __init__(self, sql_handler):
        self.sql_handler = sql_handler
        self._ensure_log_table()
    
    def _ensure_log_table(self):
        """Create logging table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            user_query TEXT,
            query_type VARCHAR(10),
            llm_responses JSON,
            sources JSON,
            feedback VARCHAR(10)
        );
        """
        self.sql_handler.execute_query(create_table_query, fetch=False)
    
    def log_interaction(self, query_log: QueryLog):
        """Log an interaction to the database"""
        insert_query = """
        INSERT INTO query_logs (timestamp, user_query, query_type, llm_responses, sources, feedback)
        VALUES (:timestamp, :user_query, :query_type, :llm_responses, :sources, :feedback);
        """
        
        log_data = query_log.to_dict()
        self.sql_handler.execute_query(insert_query, params=log_data, fetch=False)
    
    def get_logs(self, limit: int = 100):
        """Retrieve recent logs"""
        query = f"""
        SELECT * FROM query_logs 
        ORDER BY timestamp DESC 
        LIMIT {limit};
        """
        return self.sql_handler.execute_query(query, fetch=True)
    
    def update_feedback(self, timestamp: datetime, feedback: str):
        """Update feedback for a specific interaction"""
        print("timestamp in update_feedback ", timestamp)
        update_query = """
        UPDATE query_logs 
        SET feedback = :feedback 
        WHERE timestamp = :timestamp;
        """
        self.sql_handler.execute_query(
            update_query, 
            params={'feedback': feedback, 'timestamp': timestamp},
            fetch=False
        ) 