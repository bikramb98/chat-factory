import streamlit as st
from dotenv import find_dotenv, load_dotenv
import os
from sql_handler import SQLQueryHandler
from rag_handler import RAGQueryHandler
from openai import OpenAI

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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
        # print(f"called openai---------------------------- {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in OpenAI call: {e}")
        return None

def process_query(user_question, prompts):
    """Process user query and return response"""
    query_type = openai_call('gpt-4', prompts['query_type'], user_question)

    print('query type', query_type)
    
    if query_type == 'SQL':
        # Handle SQL query
        sql_handler = SQLQueryHandler()
        sql_query = openai_call('gpt-4', prompts['sql_gen'], user_question)
        
        if sql_query:
            result = sql_handler.execute_query(sql_query)
            if result:
                response = openai_call(
                    'gpt-4',
                    f"You are an expert in converting SQL query results into a sentence. {prompts['response']}",
                    f"Question: {user_question}\nSQL Results: {result}"
                )
                return {
                    'response': response,
                    'sources': [{
                        'file': 'SQL Database',
                        'similarity': 1.0,
                        'query': sql_query
                    }]
                }
            return {
                'response': "No results found for your query.",
                'sources': []
            }
    else:
        # Handle RAG query
        rag_handler = RAGQueryHandler()
        machine_name = openai_call("gpt-4", prompts['machine_name'], user_question)
        
        if machine_name:
            result = rag_handler.generate_answer(user_question, machine_name)
            return result
    
    return {
        'response': "I couldn't process your query. Please try rephrasing it or check if information exists in manual.",
        'sources': []
    }

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
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = process_query(user_question, prompts)
                
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
                
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": result['response'],
            "sources": result.get('sources', [])
        })

if __name__ == "__main__":
    main() 