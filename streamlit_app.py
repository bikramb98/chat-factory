import streamlit as st
from dotenv import find_dotenv, load_dotenv
import os
from sql_handler import SQLQueryHandler
from rag_handler import RAGQueryHandler
from openai import OpenAI
from datetime import datetime
from log_handler import LogHandler, QueryLog
import uuid
from huggingface_hub import InferenceClient

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_query_time' not in st.session_state:
    st.session_state.current_query_time = None

def load_prompts():
    """Load system prompts from files"""
    prompts = {}
    prompt_files = {
        'query_type': 'query_type_sys_prompt.txt',
        'sql_gen': 'system_prompt.txt',
        'response': 'response_sys_prompt.txt',
        'machine_name': 'machine_name_prompt.txt'
    }
    
    for key, filename in prompt_files.items():
        try:
            with open(filename, 'r') as f:
                prompts[key] = f.read()
        except FileNotFoundError:
            st.error(f"Could not find {filename}. Please ensure all prompt files are present.")
            prompts[key] = ""
    
    return prompts

def qwen_call(sys_prompt, query):

    client = InferenceClient(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    )

    response = ""
    for message in client.chat_completion(
        messages=[{"role": "system", "content": f"{sys_prompt}"},
            {"role": "user", "content": f"{query}"}],
        max_tokens=500,
        stream=True):
        content = message.choices[0].delta.content
        response += content
    return response

def llama_call(sys_prompt, query):

    client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    )

    response = ""
    for message in client.chat_completion(
        messages=[{"role": "system", "content": f"{sys_prompt}"},
            {"role": "user", "content": f"{query}"}],
        max_tokens=500,
        stream=True):
        content = message.choices[0].delta.content
        response += content

    print("Response generated by Llama . . .")
    return response

def qwen_call(sys_prompt, query):

    client = InferenceClient(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    )

    response = ""
    for message in client.chat_completion(
        messages=[{"role": "system", "content": f"{sys_prompt}"},
            {"role": "user", "content": f"{query}"}],
        max_tokens=500,
        stream=True):
        content = message.choices[0].delta.content
        response += content

    print("Response generated by Qwen . . .")
    return response

def openai_call(model, sys_prompt, user_query):
    """Make OpenAI API call"""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_TOKEN"))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7,
            max_tokens=500
        )
        print(f"OpenAI GPT4 call")
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in OpenAI call: {e}")
        return None

# def model_call_failsafe(max_attempts, action_name, primary_model, secondary_model, sys_prompt, prompt):

#     attempts = 0

#     response = None
    
#     while attempts < max_attempts:
#         response = primary_model(sys_prompt, prompt)
#         if action_name == 'query_type':
#             if response in ['RAG', 'SQL']:
#                 break  # Exit loop if valid query type is returned
#         elif action_name == 'gen_sql_query':
#             if response:
#                 break
#         attempts += 1
    
#     # If after 3 attempts the query_type is still not valid, use openai_call
#     if action_name == 'query_type':
#         if response not in ['SQL', 'RAG']:
#             response = secondary_model('gpt-4', sys_prompt, prompt)
#     elif action_name == 'gen_sql_query':
#         if not response:
#             response = secondary_model('gpt-4', sys_prompt, prompt)

#     return response

def model_call_failsafe(max_attempts, action_name, primary_model, secondary_model, sys_prompt, prompt):
    attempts = 0
    response = None
    
    while attempts < max_attempts:
        # Use the appropriate call based on the value of primary_model
        if primary_model == 'Llama':
            response = llama_call(sys_prompt, prompt)
        elif primary_model == 'GPT4':
            response = openai_call('gpt-4', sys_prompt, prompt)
        elif primary_model =='Qwen':
            response = qwen_call(sys_prompt, prompt)
        else:
            response = primary_model(sys_prompt, prompt)
        
        if action_name == 'query_type':
            if response in ['RAG', 'SQL']:
                break  # Exit loop if valid query type is returned
        elif action_name == 'gen_sql_query':
            if response:
                break
        attempts += 1
    
    # If the response is still invalid after the attempts, use secondary_model as a failsafe
    if action_name == 'query_type':
        if response not in ['SQL', 'RAG']:
            if secondary_model == 'GPT4':
                response = openai_call('gpt-4', sys_prompt, prompt)
    elif action_name == 'gen_sql_query':
        if not response:
            if secondary_model == 'GPT4':
                response = openai_call('gpt-4', sys_prompt, prompt)

    return response


def process_query(query_timestamp, user_question, prompts, log_handler):
    """Process user query and return response"""

    # Attempt to get query type using llama_call up to 3 times
    # max_attempts = 3
    # query_type = None
    # attempts = 0
    
    # while attempts < max_attempts:
    #     query_type = llama_call(prompts['query_type'], user_question)
    #     if query_type in ['SQL', 'RAG']:
    #         break  # Exit loop if valid query type is returned
    #     attempts += 1
    
    # # If after 3 attempts the query_type is still not valid, use openai_call
    # if query_type not in ['SQL', 'RAG']:
    #     query_type = openai_call('gpt-4', prompts['query_type'], user_question)

    query_type = model_call_failsafe(max_attempts=3, action_name='query_type', primary_model='Llama', secondary_model='GPT4', sys_prompt=prompts['query_type'], prompt=user_question)

    # query_type = openai_call('gpt-4', prompts['query_type'], user_question)
    # query_type = llama_call(prompts['query_type'], user_question)
    # timestamp = datetime.now()
    timestamp = query_timestamp
    
    # Initialize llm_responses dictionary to track all LLM calls
    llm_responses = {
        'query_type_response': query_type
    }

    print("Query type --> ", query_type)
    
    if query_type == 'SQL':
        # Handle SQL query
        sql_handler = SQLQueryHandler()

        # sql_query = openai_call('gpt-4', prompts['sql_gen'], user_question)
        # sql_query = qwen_call(prompts['sql_gen'], user_question)

        # Attempt to generate SQL query using qwen_call up to 3 times
        # sql_query = None
        # sql_attempts = 0
        
        # while sql_attempts < max_attempts:
        #     sql_query = qwen_call(prompts['sql_gen'], user_question)
        #     if sql_query:  # Check if a valid SQL query is generated
        #         break  # Exit loop if a valid SQL query is returned
        #     sql_attempts += 1
        
        # # If after 3 attempts the SQL query is still not valid, use openai_call
        # if not sql_query:
        #     sql_query = openai_call('gpt-4', prompts['sql_gen'], user_question)

        sql_query = model_call_failsafe(max_attempts=3, action_name='gen_sql_query', primary_model='Qwen', secondary_model='GPT4', sys_prompt=prompts['sql_gen'], prompt=user_question)

        llm_responses['sql_generation'] = sql_query

        print("sql query ", sql_query)
        
        if sql_query:
            result = sql_handler.execute_query(sql_query)
            if result:
                response = openai_call(
                    'gpt-4',
                    f"You are an expert in converting SQL query results into a sentence. {prompts['response']}",
                    f"Question: {user_question}\nSQL Results: {result}"
                )
                llm_responses['final_response'] = response
                
                sql_source = [{
                    'file': 'SQL Database',
                    'similarity': 1.0,
                    'query': sql_query
                }]
                
                # Log the interaction
                query_log = QueryLog(
                    timestamp=timestamp,
                    user_query=user_question,
                    query_type='SQL',
                    llm_responses=llm_responses,
                    sources=sql_source,
                    feedback=None  # Initialize with no feedback
                )
                log_handler.log_interaction(query_log)
                
                return {
                    'response': response,
                    'sources': sql_source
                }
            return {
                'response': "No results found for your query.",
                'sources': []
            }
    else:
        # Handle RAG query
        rag_handler = RAGQueryHandler()
        machine_name = openai_call("gpt-4", prompts['machine_name'], user_question)
        llm_responses['machine_name_detection'] = machine_name
        
        if machine_name:
            result = rag_handler.generate_answer(user_question, machine_name)
            llm_responses['final_response'] = result['response']
            
            # Log the interaction with sources
            query_log = QueryLog(
                timestamp=timestamp,
                user_query=user_question,
                query_type='RAG',
                llm_responses=llm_responses,
                sources=result['sources'],
                feedback=None  # Initialize with no feedback
            )
            log_handler.log_interaction(query_log)
            
            return result
    
    return {
        'response': "I couldn't process your query. Please try rephrasing it.",
        'sources': []
    }

def handle_feedback(feedback: str, timestamp: datetime, log_handler: LogHandler):
    """Handle user feedback"""
    print("timestamp in handle_feedback: ", timestamp)
    log_handler.update_feedback(timestamp, feedback)
    st.toast(f"Thank you for your feedback!", icon="✅")

def main():
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # Page config
    st.set_page_config(
        page_title="ChatFactory",
        page_icon="🏭",
        layout="wide"
    )
    
    # Main chat interface
    st.title("🏭 ChatFactory")
    
    # Information section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This assistant can help you with:
        - Machine maintenance queries
        - Production data analysis
        - Equipment troubleshooting
        """)
    
    # with col2:
    if st.button("Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # Initialize SQL handler and log handler
    sql_handler = SQLQueryHandler()
    log_handler = LogHandler(sql_handler)
    
    # Load prompts
    prompts = load_prompts()
    
    # Chat input
    user_question = st.chat_input("How can I assist with your manufacturing queries?")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Process new query
    if user_question:
        # Store query timestamp
        query_timestamp = datetime.now()
        st.session_state.current_query_time = query_timestamp

        print("query_timestamp is ", query_timestamp)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = process_query(query_timestamp, user_question, prompts, log_handler)
                
                # Display main response
                response_text = result['response']
                st.write(response_text)
                
                # Display source information if available
                if result['sources']:
                    with st.expander("View Source Information"):
                        st.markdown("### Reference Sources")
                        for source in result['sources']:
                            if 'query' in source:  # SQL source
                                st.markdown(f"""
                                **Source**: {source['file']}
                                **SQL Query**: ```sql
                                {source['query']}
                                ```
                                ---
                                """)
                            else:  # RAG source
                                st.markdown(f"""
                                **Source**: {source['file']}  
                                **Similarity Score**: {source['similarity']}  
                                **Chunk ID**: {source['chunk_id']}
                                ---
                                """)
                
                # Add feedback buttons
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("Correct", key=f"thumbs_up_{query_timestamp}"):
                    # if st.button("👍"):
                        st.write("button hit up")
                        print("button hit up")
                        handle_feedback("positive", query_timestamp, log_handler)
                with col2:
                    # if st.button("👎"):
                        
                    if st.button("Incorrect", key=f"thumbs_down_{query_timestamp}"):
                        st.write("button hit up")
                        print("button hit down")
                        handle_feedback("negative", query_timestamp, log_handler)
                
        # Store in chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": result['response'],
            "sources": result.get('sources', []),
            "timestamp": query_timestamp
        })

    # Add a logs viewer in the sidebar (optional)
    # with st.sidebar:
    #     if st.button("View Recent Logs"):
    #         logs = log_handler.get_logs(limit=10)
    #         st.json([dict(log) for log in logs])

if __name__ == "__main__":
    main() 