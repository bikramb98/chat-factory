import os
import PyPDF2
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import json

class RAGQueryHandler:
    def __init__(self):
        # Load configuration from config.json
        with open('configs/asset_name_config.json', 'r') as config_file:
            asset_name_config = json.load(config_file)

        self.pdf_folder = asset_name_config.get("pdf_folder")
        self.documents = []
        self.document_mappings = {}
        self.metadata_store = []
        self.asset_name = asset_name_config.get("asset_name", {})
        self.embedding_model = OpenAIEmbeddings()
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.chunk_registry = {}  # Add this to store chunk sources
        self._initialize_documents()
        
    # def _extract_text_from_pdf(self, pdf_path):
    #     """Extract text from PDF file"""
    #     text = ""
    #     with fitz.open(pdf_path) as doc:
    #         for page in doc:
    #             text += page.get_text("text") + "\n"
    #     return text

    def _extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _initialize_documents(self):
        """Initialize document embeddings"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        chunk_counter = 0
        # Process PDF files
        for file in os.listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                asset_type = self.asset_name.get(file)
                pdf_path = os.path.join(self.pdf_folder, file)
                
                extracted_text = self._extract_text_from_pdf(pdf_path)
                text_chunks = text_splitter.split_text(extracted_text)
                
                for chunk in text_chunks:
                    self.documents.append(chunk)
                    self.document_mappings[chunk_counter] = asset_type
                    self.metadata_store.append({
                        'asset_type': asset_type,
                        'source_file': file,
                        'chunk_id': chunk_counter
                    })
                    # Store chunk with source information
                    self.chunk_registry[chunk_counter] = {
                        'content': chunk,
                        'source': file,
                        'asset_type': asset_type
                    }
                    chunk_counter += 1
        
        # Create embeddings
        embeddings = [self.embedding_model.embed_query(chunk) for chunk in self.documents]
        embeddings = np.array(embeddings, dtype="float32")
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
    
    def retrieve_relevant_docs(self, query, asset_type, k=3):
        """Retrieve relevant documents with similarity scores"""
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_docs = []
        for score, idx in zip(distances[0], indices[0]):
            if self.document_mappings[idx] == asset_type:
                similarity_score = float(1 / (1 + score))  # Convert numpy.float32 to Python float
                doc_info = {
                    'content': self.documents[idx],
                    'metadata': self.metadata_store[idx],
                    'similarity_score': round(similarity_score, 3),
                    'chunk_id': int(idx)  # Convert numpy.int64 to Python int
                }
                retrieved_docs.append(doc_info)
        
        return retrieved_docs
    
    def generate_answer(self, query, asset_type):
        """Generate answer using RAG with source tracking"""
        retrieved_docs = self.retrieve_relevant_docs(query, asset_type)
        
        # Format context with source information
        context_parts = []
        sources = []
        for doc in retrieved_docs:
            context_parts.append(doc['content'])
            sources.append({
                'file': doc['metadata']['source_file'],
                'similarity': float(doc['similarity_score']),  # Ensure float type
                'chunk_id': int(doc['chunk_id'])  # Ensure int type
            })
        
        context = "\n".join(context_parts)
        
        prompt = f"""Given this context about {asset_type}:
        {context}

        Question: {query}

        Provide a clear answer and include a confidence score (0-100) for your response. 
        Format your response as:
        Answer: [Your detailed answer here]
        Confidence: [0-100]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in troubleshooting manufacturing machines. You must strictly answer based on the given context and provide a confidence score."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return {
                'response': response.choices[0].message.content,
                'sources': sources
            }
            
        except Exception as e:
            print(f"Error in generation: {e}")
            return {
                'response': "Error generating response",
                'sources': []
            } 