import os
import io
import time
import pickle
import logging
import faiss
import groq
import re  
import subprocess
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
from PyPDF2 import PdfReader
from PIL import Image
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
# Third-party imports
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from googleapiclient.http import MediaIoBaseDownload
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from typing import List, Dict, Tuple, Optional, Any
import torch
from torch.nn.functional import cosine_similarity
import os
from dotenv import load_dotenv
api_key = os.getenv("GROQ_API_KEY")


# Configuration
@dataclass
class Config:
    PDF_DOWNLOAD_DIR: Path = Path("downloaded_pdfs")
    FAISS_INDEX_DIR: Path = Path("faiss_indexes")
    TEMP_IMG_DIR: Path = Path("temp_images")
    LOG_DIR: Path = Path("logs")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    POPPLER_PATH: str = "/usr/bin"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_LENGTH: int = 4096
    OCR_THREAD_COUNT: int = max(1, min(os.cpu_count() or 4 - 1, 4))
    NUM_RETRIEVED_CHUNKS: int = 5
    REFLECTION_TEMPERATURE: float = 0.7
    RELEVANCE_THRESHOLD: float = 0.75
    MAX_ITERATIONS: int = 3
    
    def __post_init__(self):
        # Create directories
        for directory in [self.PDF_DOWNLOAD_DIR, self.FAISS_INDEX_DIR, 
                         self.TEMP_IMG_DIR, self.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize configuration
config = Config()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GoogleDriveService:
    def __init__(self):
        self.service = None
        self.initialize_service()

    def get_credentials(self) -> Optional[Credentials]:
        """Get Google Drive credentials from environment variables."""
        try:
            # Get credential components from environment variables
            client_id = os.getenv("GOOGLE_CLIENT_ID")
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
            refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
            
            # Create credentials object
            creds = Credentials(
                None,
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret
            )
            
            # Refresh to get a valid access token
            creds.refresh(Request())
            return creds
        
    except Exception as e:
        logger.error(f"Error creating credentials: {e}")
        return None

    def initialize_service(self):
        """Initialize Google Drive service."""
        creds = self.get_credentials()
        if creds:
            try:
                self.service = build("drive", "v3", credentials=creds)
                logger.info("Google Drive service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Drive service: {e}")
                self.service = None

class PDFProcessor:
    def __init__(self, embedder_model="all-MiniLM-L6-v2"):
        """Initialize the PDF Processor with an embedder model."""
        self.embedder = SentenceTransformer(embedder_model)
        # Configure tesseract with better OCR parameters
        self.custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better preservation of meaningful content."""
        if not text:
            return ""
        
        # Replace common PDF artifacts
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        # Remove non-printable characters while preserving punctuation
        text = re.sub(r'[^\x20-\x7E\n.]', '', text)
        
        # Fix common OCR mistakes
        text = re.sub(r'[|]', 'I', text)  # Replace vertical bars with 'I'
        text = re.sub(r'(?<=[0-9])/(?=[0-9])', '7', text)  # Fix '/' that should be '7'
        
        return text.strip()

    def extract_text_with_poppler(self, pdf_path: str) -> str:
        """Extract text from a PDF using Poppler with enhanced options."""
        try:
            # Use layout preservation and other optimization flags
            cmd = [
                "pdftotext",
                "-layout",  # Maintain layout
                "-nopgbrk",  # No page breaks
                "-enc", "UTF-8",  # Force UTF-8 encoding
                pdf_path,
                "-"  # Output to stdout
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return self.clean_text(result.stdout)
            else:
                logger.error(f"Poppler failed to extract text from {pdf_path}")
                return ""
        except Exception as e:
            logger.error(f"Error using Poppler: {e}")
            return ""

    def process_image(self, image: Image) -> str:
        """Enhanced image processing with better OCR settings."""
        try:
            # Preprocess image for better OCR
            image = self._preprocess_image(image)
            
            # Perform OCR with custom configuration
            text = pytesseract.image_to_string(
                image,
                lang='eng',
                config=self.custom_config
            )
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return ""

    def _preprocess_image(self, image: Image) -> Image:
        """Preprocess image for better OCR results."""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Increase contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Optional: Remove noise
            from PIL import ImageFilter
            image = image.filter(ImageFilter.MedianFilter())
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image

    def extract_text_from_pdf_images(self, pdf_path: Path) -> str:
        """Extract text from scanned PDFs using enhanced OCR."""
        try:
            # Convert PDF to images with higher DPI for better quality
            images = convert_from_path(
                str(pdf_path),
                poppler_path="/usr/bin",
                dpi=300,  # Higher DPI for better quality
                thread_count=4  # Parallel processing
            )
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                texts = list(executor.map(self.process_image, images))
            
            # Join texts with proper spacing
            return "\n\n".join(text for text in texts if text.strip())
        except Exception as e:
            logger.error(f"Error extracting text from PDF images: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Enhanced PDF text extraction with fallback mechanisms."""
        try:
            # Try PyPDF2 first
            reader = PdfReader(str(pdf_path))
            text = ""
            
            # Extract text page by page
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += page_text + "\n\n"
            
            # If PyPDF2 extraction yields limited results, try Poppler
            if len(text.split()) < 100:
                logger.info(f"Limited text from PyPDF2, trying Poppler for {pdf_path}")
                text = self.extract_text_with_poppler(str(pdf_path))
            
            # If still limited text, try OCR
            if len(text.split()) < 100:
                logger.info(f"Limited text from Poppler, using OCR for {pdf_path}")
                text = self.extract_text_from_pdf_images(pdf_path)
            
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

class FAISSManager:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks of text."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), config.CHUNK_SIZE - config.CHUNK_OVERLAP):
            chunk = " ".join(words[i:i + config.CHUNK_SIZE])
            if chunk:
                chunks.append(chunk)
        
        return chunks

    def create_index(self, chunks: List[str]) -> Tuple[faiss.Index, Dict]:
        """Create FAISS index for text chunks with dynamic clustering."""
        try:
            embeddings = np.array(self.embedder.encode(chunks, 
                                                     convert_to_tensor=False,
                                                     show_progress_bar=True)).astype(np.float32)
            dim = embeddings.shape[1]
            num_vectors = len(chunks)

            # If we have very few vectors, use a simple IndexFlatIP
            if num_vectors < 100:
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
            else:
                # For larger datasets, use IVFFlat with dynamic number of clusters
                # Rule of thumb: sqrt(n) clusters where n is number of vectors
                nlist = min(int(np.sqrt(num_vectors)), 100)  # Cap at 100 clusters
                nlist = max(1, nlist)  # Ensure at least 1 cluster
                
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

                if not index.is_trained:
                    index.train(embeddings)
                index.add(embeddings)

            metadata = {
                'chunks': chunks,
                'created_at': datetime.now().isoformat(),
                'embedding_model': config.EMBEDDING_MODEL,
                'chunk_count': len(chunks),
                'index_type': 'FlatIP' if num_vectors < 100 else f'IVFFlat(nlist={nlist})'
            }

            return index, metadata

        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise

class LLMInterface:
    def __init__(self, api_key: str):
        """Initialize Groq client with API key."""
        self.client = Groq(api_key=api_key)

    def generate_answer(self, context: str, query: str) -> str:
        """Generate answer using Groq's LLM with retry mechanism."""
        if not self.client:
            raise ValueError("Groq client not initialized")

        # Ensure context fits within model's context window
        context = context[:config.MAX_CONTEXT_LENGTH]

        # Construct the prompt with the context and query
        user_prompt = f"""Context Information:
        {context}

        Question: {query}

        Please provide a detailed response based solely on the information provided in the brochure."""

        try:
            # Call the Groq LLM to generate a response
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "system", "content": "You are a specialized educational program advisor. Provide precise answers based on the context."},
                        {"role": "user", "content": user_prompt}],
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                stream=False
            )

            # Return the generated answer
            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
                    # Exponential backoff

class DocumentProcessor:
    def __init__(self, api_key: str):
        self.drive_service = GoogleDriveService()
        self.pdf_processor = PDFProcessor()
        self.faiss_manager = FAISSManager(self.pdf_processor.embedder)
        self.llm = LLMInterface(api_key)
        self.institution_chunks = {}

    def process_folder(self, folder_id: str, institution_name: Optional[str] = None, 
                      program_name: Optional[str] = None):
        """Process all PDFs in a Google Drive folder."""
        if not self.drive_service.service:
            logger.error("Google Drive service not initialized")
            return

        try:
            # Add pageSize parameter to get all files
            query = f"'{folder_id}' in parents and trashed = false"
            results = self.drive_service.service.files().list(
                q=query,
                pageSize=1000,  # Increase page size
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
            
            items = results.get("files", [])
            logger.info(f"Found {len(items)} items in folder {folder_id}")
            
            for item in items:
                mime_type, file_name, file_id = item["mimeType"], item["name"], item["id"]
                logger.info(f"Processing item: {file_name} ({mime_type})")

                if mime_type == "application/pdf":
                    self.process_pdf(file_id, file_name, institution_name, program_name)
                elif mime_type == "application/vnd.google-apps.folder":
                    new_institution = file_name if not institution_name else institution_name
                    new_program = file_name if institution_name else None
                    self.process_folder(file_id, new_institution, new_program)

        except HttpError as error:
            logger.error(f"Error accessing Google Drive folder: {error}")

    def process_pdf(self, file_id: str, file_name: str, 
                   institution_name: Optional[str], program_name: Optional[str]):
        """Process a single PDF file."""
        try:
            logger.info(f"Processing PDF: {file_name} for {institution_name} - {program_name}")
            request = self.drive_service.service.files().get_media(fileId=file_id)
            file_path = config.PDF_DOWNLOAD_DIR / file_name

            with open(file_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            text = self.pdf_processor.extract_text_from_pdf(file_path)
            
            if text.strip():
                logger.info(f"Successfully extracted text from {file_name}")
                chunks = self.faiss_manager.create_chunks(text)
                
                if institution_name and program_name:
                    if institution_name not in self.institution_chunks:
                        self.institution_chunks[institution_name] = {}
                    if program_name not in self.institution_chunks[institution_name]:
                        self.institution_chunks[institution_name][program_name] = []
                    self.institution_chunks[institution_name][program_name].extend(chunks)
                    logger.info(f"Added {len(chunks)} chunks for {institution_name} - {program_name}")
            else:
                logger.warning(f"No text extracted from {file_name}")

        except Exception as e:
            logger.error(f"Error processing PDF {file_name}: {e}")

    def save_chunks(self):
        """Save processed chunks to file."""
        try:
            if not self.institution_chunks:
                logger.warning("No chunks to save - institution_chunks is empty")
                return
                
            logger.info(f"Saving chunks for {len(self.institution_chunks)} institutions")
            with open("institution_chunks.pkl", "wb") as f:
                pickle.dump(self.institution_chunks, f)
            logger.info("Saved institution chunks successfully")
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")

    def create_indexes(self):
        """Create FAISS indexes for all programs."""
        if not self.institution_chunks:
            logger.warning("No data to create indexes from - institution_chunks is empty")
            return
            
        for institution, programs in self.institution_chunks.items():
            for program, chunks in programs.items():
                try:
                    if not chunks:
                        logger.warning(f"No chunks found for {institution} - {program}")
                        continue
                        
                    logger.info(f"Creating index for {institution} - {program} with {len(chunks)} chunks")
                    index, metadata = self.faiss_manager.create_index(chunks)
                    
                    # Save index and metadata
                    institution_path = config.FAISS_INDEX_DIR / institution
                    institution_path.mkdir(parents=True, exist_ok=True)
                    
                    index_path = institution_path / f"{program}_index.faiss"
                    metadata_path = institution_path / f"{program}_metadata.pkl"
                    
                    faiss.write_index(index, str(index_path))
                    with open(metadata_path, "wb") as f:
                        pickle.dump(metadata, f)
                    
                    logger.info(f"Successfully created and saved index for {institution} - {program}")
                
                except Exception as e:
                    logger.error(f"Error creating index for {institution} - {program}: {e}")

def main():
    # Initialize with your API key
    processor = DocumentProcessor(api_key="")
    
    # Process files from Google Drive
    folder_id = "1P9A5ER5wa4yO2BCKzOdMA_AfDi7L8Sy_"
    
    try:
        logger.info("Starting document processing...")
        processor.process_folder(folder_id)
        
        logger.info("Saving processed chunks...")
        processor.save_chunks()
        
        logger.info("Creating FAISS indexes...")
        processor.create_indexes()
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")

if __name__ == "__main__":
    main()