from datetime import datetime
from sql_handler import SQLQueryHandler
from rag_handler import RAGQueryHandler
from log_handler import QueryLog
import json

class QueryService:
    def __init__(self, model_service, log_handler):
        self.model_service = model_service
        self.log_handler = log_handler
        self.sql_handler = SQLQueryHandler()
        self.rag_handler = RAGQueryHandler()
        self.prompts = self.load_prompts()

    def load_prompts(self):
        """Load system prompts from files"""
        prompts = {}
        
        with open('configs/sys_prompts_collection.json', 'r') as config_file:
            config = json.load(config_file)
            prompt_files = config['prompt_files']
        
        for key, filename in prompt_files.items():
            try:
                with open(filename, 'r') as f:
                    prompts[key] = f.read()
            except FileNotFoundError:
                print(f"Could not find {filename}. Please ensure all prompt files are present.")
                prompts[key] = ""
        
        return prompts

    def process_query(self, query_timestamp, user_question):
        """Process user query and return response"""
        query_type = self.model_service.openai_call('gpt-4', self.prompts['query_type'], user_question)
        timestamp = query_timestamp
        
        llm_responses = {
            'query_type_response': query_type
        }

        print("Query type --> ", query_type)
        
        if query_type == 'SQL':
            sql_query = self.model_service.openai_call('gpt-4', self.prompts['sql_gen'], user_question)
            llm_responses['sql_generation'] = sql_query

            print("sql query ", sql_query)
            
            if sql_query:
                result = self.sql_handler.execute_query(sql_query)
                if result:
                    response = self.model_service.openai_call(
                        'gpt-4',
                        self.prompts['sql_results_conversion'],
                        f"Question: {user_question}\nSQL Results: {result}"
                    )
                    llm_responses['final_response'] = response
                    
                    sql_source = [{
                        'file': 'SQL Database',
                        'similarity': 1.0,
                        'query': sql_query
                    }]
                    
                    query_log = QueryLog(
                        timestamp=timestamp,
                        user_query=user_question,
                        query_type='SQL',
                        llm_responses=llm_responses,
                        sources=sql_source,
                        feedback=None
                    )
                    self.log_handler.log_interaction(query_log)
                    
                    return {
                        'response': response,
                        'sources': sql_source
                    }
                return {
                    'response': "No results found for your query.",
                    'sources': []
                }
        else:
            asset_name = self.model_service.openai_call("gpt-4", self.prompts['asset_name'], user_question)
            llm_responses['asset_name_detection'] = asset_name
            
            if asset_name:
                result = self.rag_handler.generate_answer(user_question, asset_name)
                llm_responses['final_response'] = result['response']
                
                query_log = QueryLog(
                    timestamp=timestamp,
                    user_query=user_question,
                    query_type='RAG',
                    llm_responses=llm_responses,
                    sources=result['sources'],
                    feedback=None
                )
                self.log_handler.log_interaction(query_log)
                
                return result
        
        return {
            'response': "I couldn't process your query. Please try rephrasing it.",
            'sources': []
        } 