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
        openai_token = os.environ.get("OPENAI_API_TOKEN")
        if not openai_token:
            openai_token = st.secrets["OPENAI_API_TOKEN"]
        self.openai_client = OpenAI(api_key=openai_token)
        self.chunk_registry = {}  # Add this to store chunk sources

        # Add paths for saved data
        self.index_path = "saved_data/faiss_index.index"
        self.data_path = "saved_data/embeddings_data.json"
        
        # Try to load existing index and data, if not found, initialize from PDFs
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self._load_saved_data()
        else:
            self._initialize_documents()
            self._save_data()
            print("Initialized and saved data")        

    def _save_data(self):
        """Save FAISS index and related data to files"""
        # Create directory if it doesn't exist
        os.makedirs("saved_data", exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save related data
        data_to_save = {
            'documents': self.documents,
            'document_mappings': self.document_mappings,
            'metadata_store': self.metadata_store,
            'chunk_registry': self.chunk_registry
        }
        with open(self.data_path, 'w') as f:
            json.dump(data_to_save, f)

    def _load_saved_data(self):
        """Load FAISS index and related data from files"""
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load related data
        with open(self.data_path, 'r') as f:
            loaded_data = json.load(f)
            
        self.documents = loaded_data['documents']
        self.document_mappings = {int(k): v for k, v in loaded_data['document_mappings'].items()}
        self.metadata_store = loaded_data['metadata_store']
        self.chunk_registry = loaded_data['chunk_registry']

        print("Loaded saved data")


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
        # Check for PDF folder changes
        current_pdfs = set(f for f in os.listdir(self.pdf_folder) if f.endswith(".pdf"))
        
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                loaded_data = json.load(f)
                processed_pdfs = set(meta['source_file'] for meta in loaded_data['metadata_store'])
            
            if current_pdfs == processed_pdfs:
                self._load_saved_data()
                return

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