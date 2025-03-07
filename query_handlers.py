# query_handlers.py
from sqlalchemy import text

class SQLQueryHandler:
    def __init__(self, engine, qwen_client, sql_sys_prompt, response_sys_prompt):
        self.engine = engine
        self.qwen_client = qwen_client
        self.sql_sys_prompt = sql_sys_prompt
        self.response_sys_prompt = response_sys_prompt

    def process(self, user_question, hf_client_call):
        # Generate the SQL query using the appropriate client call
        sql_query = hf_client_call(self.qwen_client, self.sql_sys_prompt, user_question)
        if not sql_query:
            return "No SQL query generated."

        # Wrap the query with SQLAlchemy's text object
        sql_query = text(sql_query)

        try:
            with self.engine.connect() as connection:
                result = connection.execute(sql_query)
                row = result.fetchall()
        except Exception as e:
            return f"Error executing SQL query: {e}"
        finally:
            self.engine.dispose()

        # Generate a response based on the query result
        # Here, hf_client_call is used again to transform the raw SQL result into a natural language answer.
        response = hf_client_call(
            self.qwen_client,
            f"You are an expert in converting SQL query results into a sentence. Use this context: {self.response_sys_prompt}",
            f"User question: {user_question}. SQL query result: {row}"
        )
        return response

class RAGQueryHandler:
    def __init__(self, openai_client, embedding_model, documents, document_mappings):
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.documents = documents
        self.document_mappings = document_mappings

    def retrieve_relevant_docs(self, query, machine_type, k=3):
        # Get the query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = [query_embedding]

        # For demonstration, assume `index` is built externally and available as a global
        # You could also pass the FAISS index in the constructor
        from faiss import IndexFlatL2  # Assuming FAISS is set up similarly as in your original code
        d = self.documents[0].shape[0]  # Example: assume each embedding has same length
        index = IndexFlatL2(d)
        index.add(self.documents)
        distances, indices = index.search(query_embedding, k)

        filtered_results = []
        for i in indices[0]:
            if self.document_mappings.get(i) == machine_type:
                filtered_results.append(self.documents[i])
            if len(filtered_results) >= k:
                break
        return filtered_results

    def process(self, user_question, machine_type):
        # Retrieve documents relevant to the machine type
        retrieved_docs = self.retrieve_relevant_docs(user_question, machine_type)
        context = "\n".join(retrieved_docs)

        prompt = f"""Given this context about {machine_type}:
{context}

Question: {user_question}

Provide a clear answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in troubleshooting manufacturing machines. Strictly answer only based on the given context. If not found, respond with 'I don't know the answer based on the available information.'"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"