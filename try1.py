import os
import io
import time
import pickle
import logging
import faiss
import groq
import os
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
        """Get and refresh Google Drive credentials."""
        creds = None
        token_path = Path("token.pickle")
        
        if token_path.exists():
            try:
                with open(token_path, "rb") as token:
                    creds = pickle.load(token)
            except Exception as e:
                logger.error(f"Error loading credentials: {e}")
                return None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    with open(token_path, "wb") as token:
                        pickle.dump(creds, token)
                except RefreshError:
                    logger.error("Failed to refresh credentials")
                    return None
            else:
                logger.error("No valid credentials available")
                return None

        return creds

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

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra spaces and special characters."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()

    def extract_text_with_poppler(self, pdf_path: str) -> str:
        """Extract text from a PDF using Poppler's pdftotext."""
        try:
            result = subprocess.run(["pdftotext", pdf_path, "-"], capture_output=True, text=True)
            if result.returncode == 0:
                return self.clean_text(result.stdout)
            else:
                logger.error(f"Poppler failed to extract text from {pdf_path}")
                return ""
        except Exception as e:
            logger.error(f"Error using Poppler: {e}")
            return ""

    def process_image(self, image: Image) -> str:
        """Process a single image using OCR."""
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return ""

    def extract_text_from_pdf_images(self, pdf_path: Path) -> str:
        """Extract text from scanned PDFs using OCR."""
        try:
            images = convert_from_path(str(pdf_path), poppler_path="/usr/bin")
            with ThreadPoolExecutor(max_workers=4) as executor:
                texts = list(executor.map(self.process_image, images))
            return "\n\n".join(text for text in texts if text.strip())
        except Exception as e:
            logger.error(f"Error extracting text from PDF images: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF using both native extraction and OCR."""
        try:
            reader = PdfReader(str(pdf_path))
            text = ""

            # Try native text extraction first
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += page_text + "\n\n"

            # If native extraction fails, use OCR
            if len(text.split()) < 100:
                logger.info(f"Limited text extracted from {pdf_path}, using OCR...")
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

    def generate_answer(self, context: str, query: str, max_retries: int = 3) -> str:
        """Generate answer using Groq's LLM with retry mechanism."""
        if not self.client:
            raise ValueError("Groq client not initialized")

        # Ensure context fits within model's context window
        context = context[:config.MAX_CONTEXT_LENGTH]

        # Create a more focused system prompt
        system_prompt = """You are a specialized educational program advisor. Your role is to:
        1. Provide precise and direct answers based on the given context.
        2. Focus on answering the specific question asked by the user.
        3. Do not explain or reason unless necessary.
        4. Maintain a professional and helpful tone."""

        # Construct the user prompt
        user_prompt = f"""Context Information:
        {context}

        Question: {query}

        Please provide a detailed response based solely on the information provided in the context."""

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}],
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                stream=False
            )
               # Return the answer without unnecessary reasoning
        answer = completion.choices[0].message.content
        return answer.strip()

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate answer after {max_retries} attempts: {e}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff


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

            # Clean up downloaded file
            os.remove(file_path)

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
