import streamlit as st
from dotenv import find_dotenv, load_dotenv
from sql_handler import SQLQueryHandler
from log_handler import LogHandler
from datetime import datetime
from services.model_service import ModelService
from services.query_service import QueryService

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_query_time' not in st.session_state:
    st.session_state.current_query_time = None

# Initialize services only once using session state
if 'services' not in st.session_state:
    load_dotenv(find_dotenv())
    sql_handler = SQLQueryHandler()
    log_handler = LogHandler(sql_handler)
    model_service = ModelService()
    query_service = QueryService(model_service, log_handler)
    st.session_state.services = {
        'sql_handler': sql_handler,
        'log_handler': log_handler,
        'model_service': model_service,
        'query_service': query_service
    }

def handle_feedback(feedback: str, timestamp: datetime, log_handler: LogHandler):
    """Handle user feedback"""
    print("timestamp in handle_feedback: ", timestamp)
    log_handler.update_feedback(timestamp, feedback)
    st.toast(f"Thank you for your feedback!", icon="✅")

def main():
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
    
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Get services from session state
    query_service = st.session_state.services['query_service']
    log_handler = st.session_state.services['log_handler']
    
    # Chat input
    user_question = st.chat_input("How can I assist with your manufacturing queries?")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Process new query
    if user_question:
        query_timestamp = datetime.now()
        st.session_state.current_query_time = query_timestamp
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_service.process_query(query_timestamp, user_question)
                
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
                        handle_feedback("positive", query_timestamp, log_handler)
                with col2:
                    if st.button("Incorrect", key=f"thumbs_down_{query_timestamp}"):
                        handle_feedback("negative", query_timestamp, log_handler)
                
        # Store in chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": result['response'],
            "sources": result.get('sources', []),
            "timestamp": query_timestamp
        })

if __name__ == "__main__":
    main() 