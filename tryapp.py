import streamlit as st
import os
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st

# Load from .env file for local testing
load_dotenv()

# For Streamlit Cloud - get secrets
if hasattr(st, 'secrets'):
    # Copy secrets to environment variables
    if 'GOOGLE_CLIENT_ID' in st.secrets:
        os.environ['GOOGLE_CLIENT_ID'] = st.secrets['GOOGLE_CLIENT_ID']
    if 'GOOGLE_CLIENT_SECRET' in st.secrets:
        os.environ['GOOGLE_CLIENT_SECRET'] = st.secrets['GOOGLE_CLIENT_SECRET']
    if 'GOOGLE_REFRESH_TOKEN' in st.secrets:
        os.environ['GOOGLE_REFRESH_TOKEN'] = st.secrets['GOOGLE_REFRESH_TOKEN']
    if 'GROQ_API_KEY' in st.secrets:
        os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
        
# Configuration
@dataclass
class Config:
    PDF_DOWNLOAD_DIR: Path = Path("downloaded_pdfs")
    FAISS_INDEX_DIR: Path = Path("faiss_indexes")
    TEMP_IMG_DIR: Path = Path("temp_images")
    LOG_DIR: Path = Path("logs")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_CONTEXT_LENGTH: int = 4096
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

# API Configuration for different providers
API_CONFIGS = {
    "Groq - llama3-70b-8192": {
        "base_url": "https://api.groq.com/openai/v1",  # Corrected URL format
        "model": "llama3-70b-8192",
        "api_key_prefix": "groq"
    },
    "OpenAI - GPT-4": {  # Corrected model name
        "base_url": "https://api.openai.com/v1",  # Removed trailing slash
        "model": "gpt-4",  # Corrected model name
        "api_key_prefix": "openai"
    },
    "OpenAI - GPT-3.5 Turbo": {
        "base_url": "https://api.openai.com/v1",  # Removed trailing slash
        "model": "gpt-3.5-turbo",
        "api_key_prefix": "openai"
    },
    "Groq - Deepseek-R1-70B": {  # Fixed spacing and capitalization
        "base_url": "https://api.groq.com/openai/v1",  # Corrected URL format
        "model": "deepseek-r1-distill-llama-70b",
        "api_key_prefix": "groq"
    }
}

class QASystem:
    def __init__(self, api_keys: Dict[str, str], selected_model: str):
        """Initialize the QA system with both embedding model and LLM client."""
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.selected_model = selected_model
        self.api_config = API_CONFIGS[selected_model]
        
        # Initialize the appropriate client based on the selected model
        api_key = api_keys.get(self.api_config["api_key_prefix"])
        if not api_key:
            raise ValueError(f"API key not found for {self.api_config['api_key_prefix']}")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.api_config["base_url"]
        )
        self.institutions = self._load_available_institutions()
        self.load_institutions_data()

    def _load_available_institutions(self) -> Dict[str, List[str]]:
        """Load available institutions and their programs from FAISS index directory."""
        institutions = {}
        if Config.FAISS_INDEX_DIR.exists():
            for inst_dir in Config.FAISS_INDEX_DIR.iterdir():
                if inst_dir.is_dir():
                    programs = []
                    for file in inst_dir.glob("*_index.faiss"):
                        program_name = file.stem.replace("_index", "")
                        programs.append(program_name)
                    if programs:
                        institutions[inst_dir.name] = sorted(programs)
        return institutions

    def load_institutions_data(self):
        """Load saved institution chunks for additional context."""
        try:
            with open("institution_chunks.pkl", "rb") as f:
                self.institution_chunks = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load institution chunks: {e}")
            self.institution_chunks = {}

    def search_similar_chunks(self, institution: str, program: str, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant text chunks using FAISS with enhanced error handling."""
        try:
            index_path = Config.FAISS_INDEX_DIR / institution / f"{program}_index.faiss"
            metadata_path = Config.FAISS_INDEX_DIR / institution / f"{program}_metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                st.error(f"Index files not found for {institution} - {program}")
                return []

            index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            query_vector = np.array([self.embedder.encode(query)]).astype(np.float32)
            D, I = index.search(query_vector, top_k)
            
            relevant_chunks = []
            for score, idx in zip(D[0], I[0]):
                if idx < len(metadata['chunks']) and score > 0.3:
                    relevant_chunks.append(metadata['chunks'][idx])
            
            return relevant_chunks

        except Exception as e:
            st.error(f"Error searching index: {e}")
            return []

    def generate_answer(self, context: str, query: str) -> str:
        """Generate answer using the selected LLM with retry mechanism."""
        if not self.client:
            raise ValueError("LLM client not initialized")

        context = context[:Config.MAX_CONTEXT_LENGTH]

        user_prompt = f"""Brochure Information:
        {context}

        Question: {query}

        Please provide a detailed response based solely on the information provided in the Brochure."""

        try:
            # Adjust parameters based on the model provider
            if "groq" in self.api_config["api_key_prefix"]:
                completion = self.client.chat.completions.create(
                    model=self.api_config["model"],
                    messages=[
                        {"role": "system", "content": "You are a specialized educational program advisor. Provide precise answers based on the brochure."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.9,
                    stream=False,
                    stop=None  
                )
            else:
                completion = self.client.chat.completions.create(
                    model=self.api_config["model"],
                    messages=[
                        {"role": "system", "content": "You are a specialized educational program advisor. Provide precise answers based on the brochure."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.9,
                    stream=False
                )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            st.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer at this time."

def main():
    st.set_page_config(
        page_title="Educational Program Assistant",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Brochure Query Assistant")
    st.markdown("""
    Explore educational programs across different institutions. Get detailed information about program 
    details, requirements, curriculum, and more! Just select an institution, choose a program, and ask 
    your questions.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API key inputs for different providers
        api_keys = {}
        st.subheader("API Keys")
        groq_key = st.text_input("Enter Groq API Key:", type="password", 
                                help="Required for Groq models")
        openai_key = st.text_input("Enter OpenAI API Key:", type="password",
                                  help="Required for OpenAI models")
        if groq_key:
            api_keys["groq"] = groq_key
        if openai_key:
            api_keys["openai"] = openai_key

        # Model selection with grouping
        st.subheader("Model Selection")
        
        # Group models by provider for cleaner selection
        providers = {
            "Groq Models": [k for k in API_CONFIGS.keys() if k.startswith("Groq ")],  # Added space after Groq
            "OpenAI Models": [k for k in API_CONFIGS.keys() if k.startswith("OpenAI")]
        }
        
        # First select provider, then model
        selected_provider = st.selectbox(
            "Select Provider:",
            options=list(providers.keys())
        )
        
        selected_model = st.selectbox(
            "Select Model:",
            options=providers[selected_provider],
            help="Choose the AI model to use for generating answers"
        )
        
        st.markdown("---")
        st.header("📖 How to Use")
        st.markdown("""
        1. Enter the required API keys
        2. Select your preferred AI provider and model
        3. Choose an institution and program
        4. Type your question
        5. Click "Get Answer" for detailed information
        
        💡 **Tips:**
        - Different models may provide varying responses
        - Ensure you have the appropriate API key for your selected model
        """)
        
        #st.markdown("---")
        #st.header("🛠️ Advanced Settings")
        top_k = 5 #st.slider("Number of relevant chunks to consider:", 1, 5, 3,
                         #help="Higher numbers may provide more context but could be less focused")

    # Check for required API key based on selected model
    required_key_prefix = API_CONFIGS[selected_model]["api_key_prefix"]
    if required_key_prefix not in api_keys:
        st.warning(f"Please enter the {required_key_prefix.title()} API key to use the selected model.")
        return

    # Initialize QA system
    try:
        qa_system = QASystem(api_keys, selected_model)
    except Exception as e:
        st.error(f"Error initializing QA system: {e}")
        return

    if not qa_system.institutions:
        st.error("No institution data found. Please ensure the FAISS indexes are properly created.")
        return

    # Main interface
    col1, col2 = st.columns([1, 2])

    with col1:
        institution = st.selectbox(
            "Select Institution:",
            options=sorted(qa_system.institutions.keys())
        )

        if institution:
            program = st.selectbox(
                "Select Program:",
                options=qa_system.institutions[institution]
            )

        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What are the admission requirements? What is the program duration?"
        )

        search_button = st.button("🔍 Get Answer", type="primary")

    if search_button and question and institution and program:
        with col2:
            progress_bar = st.progress(0)
            
            with st.spinner("🔍 Searching through program documentation..."):
                relevant_chunks = qa_system.search_similar_chunks(
                    institution, program, question, top_k
                )
                progress_bar.progress(50)
                
                if not relevant_chunks:
                    st.error("No relevant information found for your question.")
                    return

            with st.spinner("💭 Generating comprehensive answer..."):
                context = "\n".join(relevant_chunks)
                answer = qa_system.generate_answer(context, question)
                progress_bar.progress(100)

            st.success("✨ Answer generated successfully!")
            
            st.markdown("### 📝 Answer")
            st.markdown(answer)

            with st.expander("🔍 View Source Context"):
                st.markdown("### Referenced Program Documentation")
                for i, chunk in enumerate(relevant_chunks, 1):
                    with st.container():
                        st.markdown(f"**Excerpt {i}:**")
                        st.markdown(chunk)
                        st.divider()

if __name__ == "__main__":
    main()