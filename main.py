from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient
import os
import streamlit as st
import re
import pymysql
from sqlalchemy import create_engine, text
import json

user_question = 'Which machine was down for the longest this week and how long in total?'

# What was the last machine to have belt misalignment?
# Which shift produced the most number of parts today?
# Which machine was down for the longest this week and how long in total?

# Database connection details
with open("db_config.json", "r") as file:
    config = json.load(file)

# Extract details
DB_USER = config["DB_USER"]
DB_PASSWORD = config["DB_PASSWORD"]
DB_NAME = config["DB_NAME"]
PUBLIC_IP = config["PUBLIC_IP"]
PORT = config["PORT"]
# Create connection string
connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{PUBLIC_IP}:{PORT}/{DB_NAME}"

# Create engine
engine = create_engine(connection_url)

#open and read system_prompt.txt file
sys_prompt_file = open('system_prompt.txt', 'r')
sys_prompt = sys_prompt_file.read()

response_prompt_file = open('response_sys_prompt.txt', 'r') 
response_prompt = response_prompt_file.read()

# print(sys_prompt)

load_dotenv(find_dotenv())

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# client = InferenceClient(
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     token=HUGGINGFACE_API_TOKEN,
# )

client = InferenceClient(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    token=HUGGINGFACE_API_TOKEN,
)

full_response = ""

# print(sys_prompt)


for message in client.chat_completion(
    messages=[{"role": "system", "content": f"{sys_prompt}"},
        {"role": "user", "content": f"{user_question}"}],
    max_tokens=500,
    stream=True):

    content = message.choices[0].delta.content
    full_response += content

print(full_response)

# full_response = text("""SELECT machine_id, machine_name, last_downtime, downtime_duration, issue_description
# FROM machine_status
# WHERE issue_description = 'Belt misalignment'
# ORDER BY last_downtime DESC
# LIMIT 1;""")

if full_response != "":

    sql_query = text(f"""{full_response}""")

    # Execute the query and fetch the result
    try:
        with engine.connect() as connection:
            result = connection.execute(sql_query)
            row = result.fetchone()  # Fetch the first row
            print(row)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        engine.dispose()  # Close the database connection


    full_response = ""

    for message in client.chat_completion(
        messages=[{"role": "system", "content": f"You are an expert in converting a SQL query results into a sentence, with the knowledge of what the query is about. Only response back with the sentence, generated by using the SQL query response and the context that generated it. You also have the following informaiton: {response_prompt}"},
            {"role": "user", "content": f"The question that the user asked is {user_question}. The column values are given in {sys_prompt}. The SQL query response is {row}. Add knowledge from your system prompt regarding the various columns in the table. Frame a sentence to communicate this to the user."}],
        max_tokens=500,
        stream=True):

        content = message.choices[0].delta.content
        full_response += content

    print(full_response)