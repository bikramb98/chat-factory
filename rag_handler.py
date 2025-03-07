import os
import fitz
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

class RAGQueryHandler:
    def __init__(self, pdf_folder="machine_manuals"):
        self.pdf_folder = pdf_folder
        self.documents = []
        self.document_mappings = {}
        self.metadata_store = []
        self.machine_names = {
            "cnc_guide.pdf": "CNC Lathe-1000",
            "hydraulic_press_guide.pdf": "Hydraulic Press-200",
            "laser_cutter_guide.pdf": "Laser Cutter-X5"
        }
        self.embedding_model = OpenAIEmbeddings()
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._initialize_documents()
        
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text
    
    def _initialize_documents(self):
        """Initialize document embeddings"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Process PDF files
        for file in os.listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                machine_type = self.machine_names.get(file)
                pdf_path = os.path.join(self.pdf_folder, file)
                
                extracted_text = self._extract_text_from_pdf(pdf_path)
                text_chunks = extracted_text.split("\n\n")
                
                for chunk in text_chunks:
                    self.documents.append(chunk)
                    self.document_mappings[len(self.metadata_store)] = machine_type
                    self.metadata_store.append(machine_type)
        
        # Create embeddings
        embeddings = [self.embedding_model.embed_query(chunk) for chunk in self.documents]
        embeddings = np.array(embeddings, dtype="float32")
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
    
    def retrieve_relevant_docs(self, query, machine_type, k=3):
        """Retrieve relevant documents for the query"""
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_embedding, k)
        
        filtered_results = []
        for i in indices[0]:
            if self.document_mappings[i] == machine_type:
                filtered_results.append(self.documents[i])
            if len(filtered_results) >= k:
                break
        return filtered_results
    
    def generate_answer(self, query, machine_type):
        """Generate answer using RAG"""
        retrieved_docs = self.retrieve_relevant_docs(query, machine_type)
        context = "\n".join(retrieved_docs)
        
        prompt = f"""Given this context about {machine_type}:
        {context}

        Question: {query}

        Provide a clear answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in troubleshooting manufacturing machines. You must strictly answer only based on the given context. If the answer is not found in the context, say: 'I don't know the answer based on the available information.' Do not infer anything beyond the context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generation: {e}")
            return "Error generating response" 