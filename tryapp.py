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

class QASystem:
    def __init__(self, api_key: str):
        """Initialize the QA system with both embedding model and LLM client."""
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
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
            
            # Filter out irrelevant results based on similarity score
            relevant_chunks = []
            for score, idx in zip(D[0], I[0]):
                if idx < len(metadata['chunks']) and score > 0.3:  # Similarity threshold
                    relevant_chunks.append(metadata['chunks'][idx])
            
            return relevant_chunks

        except Exception as e:
            st.error(f"Error searching index: {e}")
            return []

    def generate_answer(self, context: str, query: str) -> str:
        """Generate answer using LLM with enhanced prompt."""
        try:
            context = context[:Config.MAX_CONTEXT_LENGTH]
            
            system_prompt = """You are a specialized educational program advisor. Your role is to:
            1. Provide accurate information based solely on the given context
            2. Focus on program details, requirements, and key features
            3. Clearly indicate if certain information is not available in the context
            4. Use bullet points for lists and structured information
            5. Maintain a professional and helpful tone
            6. Cite specific parts of the context when possible"""

            user_prompt = f"""Context Information:
            {context}

            Question: {query}

            Please provide a detailed response based solely on the information provided in the context."""

            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[ 
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer at this time."

def main():
    st.set_page_config(
        page_title="Educational Program Assistant",
        page_icon="🎓",
        layout="wide"
    )

    # Title and description
    st.title("🎓 Educational Program Information Assistant")
    st.markdown("""
    Explore educational programs across different institutions. Get detailed information about program 
    details, requirements, curriculum, and more! Just select an institution, choose a program, and ask 
    your questions.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Enter Groq API Key:", type="password", 
                               help="Your Groq API key is required for generating answers")
        
        st.markdown("---")
        st.header("📖 How to Use")
        st.markdown("""
        1. Enter your Groq API key
        2. Select an institution from the dropdown
        3. Choose a specific program
        4. Type your question in the text area
        5. Click "Get Answer" to receive detailed information
        
        💡 **Tips:**
        - Ask specific questions for better results
        - You can view source context for transparency
        - The system searches through program documentation
        """)
        
        # Additional settings
        st.markdown("---")
        st.header("🛠️ Advanced Settings")
        top_k = st.slider("Number of relevant chunks to consider:", 1, 5, 3,
                         help="Higher numbers may provide more context but could be less focused")

    if not api_key:
        st.warning("Please enter your Groq API key to continue.")
        return

    # Initialize QA system
    qa_system = QASystem(api_key)

    if not qa_system.institutions:
        st.error("No institution data found. Please ensure the FAISS indexes are properly created.")
        return

    # Main interface with two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Institution and program selection
        institution = st.selectbox(
            "Select Institution:",
            options=sorted(qa_system.institutions.keys())
        )

        if institution:
            program = st.selectbox(
                "Select Program:",
                options=qa_system.institutions[institution]
            )

        # Question input with placeholder
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What are the admission requirements? What is the program duration?"
        )

        # Search button with loading state
        search_button = st.button("🔍 Get Answer", type="primary")

    # Results display
    if search_button and question and institution and program:
        with col2:
            # Progress tracking
            progress_bar = st.progress(0)
            
            # Search phase
            with st.spinner("🔍 Searching through program documentation..."):
                relevant_chunks = qa_system.search_similar_chunks(
                    institution, program, question, top_k
                )
                progress_bar.progress(50)
                
                if not relevant_chunks:
                    st.error("No relevant information found for your question.")
                    return

            # Answer generation phase
            with st.spinner("💭 Generating comprehensive answer..."):
                context = "\n".join(relevant_chunks)
                answer = qa_system.generate_answer(context, question)
                progress_bar.progress(100)

            # Display results
            st.success("✨ Answer generated successfully!")
            
            # Answer display
            st.markdown("### 📝 Answer")
            st.markdown(answer)

            # Source context
            with st.expander("🔍 View Source Context"):
                st.markdown("### Referenced Program Documentation")
                for i, chunk in enumerate(relevant_chunks, 1):
                    with st.container():
                        st.markdown(f"**Excerpt {i}:**")
                        st.markdown(chunk)
                        st.divider()

            # Feedback buttons
            col1, col2, col3 = st.columns([1,1,3])
            with col1:
                st.button("👍 Helpful")
            with col2:
                st.button("👎 Not Helpful")
            with col3:
                st.caption("Your feedback helps us improve!")

if __name__ == "__main__":
    main()
