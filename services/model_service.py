from openai import OpenAI
from huggingface_hub import InferenceClient
import os
import streamlit as st

class ModelService:
    def __init__(self):

        openai_token = os.environ.get("OPENAI_API_TOKEN")
        if not openai_token:
            openai_token = st.secrets["OPENAI_API_TOKEN"]

        self.openai_client = OpenAI(api_key=openai_token)
    
    def qwen_call(self, sys_prompt, query):

        token = os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            token = st.secrets["HUGGINGFACE_TOKEN"]

        client = InferenceClient(
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            token=token,
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

    def llama_call(self, sys_prompt, query):

        token = os.environ.get("HUGGINGFACE_TOKEN")     
        if not token:
            token = st.secrets["HUGGINGFACE_TOKEN"]

        client = InferenceClient(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token=token,
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

    def openai_call(self, model, sys_prompt, user_query):
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error in OpenAI call: {e}")
            return None

    def model_call_failsafe(self, max_attempts, action_name, primary_model, secondary_model, sys_prompt, prompt):
        attempts = 0
        response = None
        
        while attempts < max_attempts:
            if primary_model == 'Llama':
                response = self.llama_call(sys_prompt, prompt)
            elif primary_model == 'GPT4':
                response = self.openai_call('gpt-4', sys_prompt, prompt)
            elif primary_model == 'Qwen':
                response = self.qwen_call(sys_prompt, prompt)
            
            if action_name == 'query_type':
                if response in ['RAG', 'SQL']:
                    break
            elif action_name == 'gen_sql_query':
                if response:
                    break
            attempts += 1
        
        if action_name == 'query_type':
            if response not in ['SQL', 'RAG']:
                if secondary_model == 'GPT4':
                    response = self.openai_call('gpt-4', sys_prompt, prompt)
        elif action_name == 'gen_sql_query':
            if not response:
                if secondary_model == 'GPT4':
                    response = self.openai_call('gpt-4', sys_prompt, prompt)

        return response 