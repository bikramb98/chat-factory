from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient
import os
import streamlit as st
import re
import pymysql
from sqlalchemy import create_engine, text
import json
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import fitz
import faiss
import numpy as np
from openai import OpenAI

#TODO: Classify user question to either query SQL or RAG

user_question = 'When was pneumatic cylinders last updated in inventory?'

def get_query_type(user_question):

    query_type = ""

    for message in llama_client.chat_completion(
        messages=[{"role": "system", "content": f"{query_type_sys_prompt}"},
            {"role": "user", "content": f"{user_question}"}],
        max_tokens=500,
        stream=True):

        content = message.choices[0].delta.content
        query_type += content
    
    return query_type

    # print(f"sys prompt: {query_type_sys_prompt}")

    # response = openai_client.chat.completions.create(
    # model="gpt-4",
    # messages=[
    #     {"role": "system", "content": f"{query_type_sys_prompt}"},
    #     {"role": "user", "content": f"{user_question}"}
    # ],
    # temperature=0.7,
    # max_tokens=500
    #     )

    # return response

# Which machine was down for the longest last last week and how long in total?
# How long was each machine down this week/ last week?
# What was the last machine to have belt misalignment?
# Which shift produced the most number of parts today?
# Which machine was down for the longest this week and how long in total?
# Which shift produced the most parts last week? What was the date? - change one units_produced to 245 or somet, as all others are 240
# How many instances of Coolant Leakage in the last week?
# Which was the most frequently down machine last week?
# Which machine was down for the longest this week and what was the downtime reason
# How many hydraulic seals are left and where are they?
# When was pneumatic cylinders last updated in inventory?

# Which machine downtime had the most impact on units produced last week?
# The most impactful machine downtime on units produced last week was due to the machine named CNC Lathe-1000, which had a downtime duration of 120 minutes due to excessive vibration, affecting the production on 2023-10-02.

# Not working: 
# Which shift in a particular day produced the most parts last week?

#TO try:
# 
#

# Agents questions:
# 'Which machine downtime had the most impact on units produced last week?'

# RAG questions:
# My Laser Cutter-X5 has lost its paint shine. Why?


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

machine_name_sys_prompt_file = open('machine_name_prompt.txt', 'r')
machine_name_sys_prompt = machine_name_sys_prompt_file.read()

query_type_sys_prompt_file = open('query_type_sys_prompt.txt', 'r')
query_type_sys_prompt = query_type_sys_prompt_file.read()

# print(sys_prompt)

load_dotenv(find_dotenv())

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

llama_client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=HUGGINGFACE_API_TOKEN,
)

client = InferenceClient(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    token=HUGGINGFACE_API_TOKEN,
)

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

query_type = get_query_type(user_question)

print(f'QUERY TYPE IS -------> {query_type}')

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
            row = result.fetchall()  # Fetch the first row
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

###############xxxxxxxxxxx##############

# openai.api_key = os.getenv('OPENAIKEY')

query = 'Why is tool misalignment in the laser cutter?'

# Step 1: Load PDF and Extract Text
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# Step 2: Chunk the text for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Size of each chunk
    chunk_overlap=100  # Overlapping for better context retention
)

# Load PDF files and extract text
documents = []
document_mappings = {}  # Store which chunks belong to which machine
metadata_store = []  # Stores metadata (machine type) corresponding to each embedding
chunk_registry = {}
chunk_counter = 0

machine_names = {
    "cnc_guide.pdf": "CNC Lathe-1000",
    "hydraulic_press_guide.pdf": "Hydraulic Press-200",
    "laser_cutter_guide.pdf": "Laser Cutter-X5"
}

pdf_folder = "machine_manuals"  # Change this to the folder containing PDFs
# for file in os.listdir(pdf_folder):
#     if file.endswith(".pdf"):
#         pdf_path = os.path.join(pdf_folder, file)
#         extracted_text = extract_text_from_pdf(pdf_path)
#         chunks = text_splitter.split_text(extracted_text)
#         documents.extend(chunks)

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        machine_type = machine_names.get(file)  # Get machine type from filename
        pdf_path = os.path.join(pdf_folder, file)
        
        extracted_text = extract_text_from_pdf(pdf_path)
        text_chunks = extracted_text.split("\n\n")  # Basic chunking
        
        for chunk in text_chunks:
            documents.append(chunk)
            document_mappings[len(metadata_store)] = machine_type  # Store chunk index with machine type
            metadata_store.append(machine_type)

# Step 3: Convert documents into embeddings
embedding_model = OpenAIEmbeddings()
embeddings = [embedding_model.embed_query(chunk) for chunk in documents]
embeddings = np.array(embeddings, dtype="float32")

# Step 3: Convert documents into embeddings and store them
# embeddings = np.array([embedding_model.encode(doc) for doc in documents], dtype="float32")
# index.add(embeddings)

# Step 4: Store embeddings in FAISS index
d = embeddings.shape[1]  # Vector dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Step 5: Query Processing & Retrieval
def retrieve_relevant_docs(query, machine_type, k=3):
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_embedding, k)
    filtered_results = []
    for i in indices[0]:
        print(document_mappings[i])
        if document_mappings[i] == machine_type:
            filtered_results.append(documents[i])
        if len(filtered_results) >= k:
            break  # Stop once we have enough filtered results
    
    return filtered_results
    # retrieved_docs = [documents[i] for i in indices[0]]
    # return retrieved_docs

# def retrieve_relevant_docs(query, machine_type, k=3):
#     query_embedding = np.array([embedding_model.encode(query)], dtype="float32")
#     distances, indices = index.search(query_embedding, k * 2)  # Retrieve more to filter
    
#     filtered_results = []
#     for i in indices[0]:
#         if document_mappings[i] == machine_type:
#             filtered_results.append(documents[i])
#         if len(filtered_results) >= k:
#             break  # Stop once we have enough filtered results
    
    # return filtered_results

# # Step 6: Pass Retrieved Context to LLM
# def generate_answer_with_rag(query, machine_type):


#     retrieved_docs = retrieve_relevant_docs(query, machine_type)
#     context = "\n".join(retrieved_docs)

#     print(f"context retireved is {context}")
#     prompt = f"""
#     You are a manufacturing troubleshooting assistant. 
#     Use the following extracted document content to answer the user query:

#     Machine: {machine_type}
    
#     Context: {context}
    
#     User Query: {query}
    
#     Provide a clear, step-by-step answer in English:
#     """
#     # response = openai.ChatCompletion.create(
#     #     model="gpt-4",
#     #     messages=[{"role": "system", "content": "You are an expert in troubleshooting manufacturing machines."},
#     #               {"role": "user", "content": prompt}]
#     # )


#     #####
#     full_response = ""

#     for message in llama_client.chat_completion(
#         messages=[
#                   {"role": "user", "content": 'hi who are you?'}],
#         max_tokens=500,
#         stream=True):

#         # print(message.choices[0].delta.content)

#         if message.choices[0].delta.content is not None:  # Add this check
#             full_response += message.choices[0].delta.content

#         # content = message.choices[0].delta.content
#         # full_response += content

#     # print(full_response)

#     return full_response

#     #####
#     # return response["choices"][0]["message"]["content"]



def generate_answer_with_rag(query, machine_type):
    retrieved_docs = retrieve_relevant_docs(query, machine_type)
    context = "\n".join(retrieved_docs)

    prompt = f"""Given this context about {machine_type}:
{context}

Question: {query}

Provide a clear answer:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in troubleshooting manufacturing machines. If you do not know the answer, say that you don't know, rather than returning false information"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generation: {e}")
        return "Error generating response"

# Example Query
test_query = "The laser cutter has lost its paint shine. Why?"


# Get machine query

def get_machine_name(test_query):

    machine_name = ""

    for message in llama_client.chat_completion(
        messages=[{"role": "system", "content": f"{machine_name_sys_prompt}"},
            {"role": "user", "content": f"{test_query}"}],
        max_tokens=500,
        stream=True):

        content = message.choices[0].delta.content
        machine_name += content

    return machine_name



# machine_query = "Laser Cutter-X5"
machine_query = get_machine_name(test_query)

print(f"Machine query from llama is ------> {machine_query}")

answer = generate_answer_with_rag(test_query, machine_query)
print("\nGenerated Response:", answer)