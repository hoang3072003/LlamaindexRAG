import os
import tempfile
import shutil
import pdfplumber
import io
import base64
import logging
from typing import List, Any
from PIL import Image
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

GOOGLE_API_KEY = "AIzaSyCVSFxhcx5B-b4pmW1Ywy1TEoK1xOGTbjg"
Settings.llm = Gemini(api_key=GOOGLE_API_KEY)
logger = logging.getLogger(__name__)

def create_vector_db(file_upload, chroma_client) -> VectorStoreIndex:
    """
    Create a vector database from a PDF file upload.
    """
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file_upload.name)

    # Save uploaded file to temporary directory
    with open(pdf_path, "wb") as f:
        f.write(file_upload.getvalue())

    # Load data from the saved PDF
    loader = SimpleDirectoryReader(input_dir=temp_dir)
    documents = loader.load_data()

    # Set up embedding model and vector store
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    chroma_collection = chroma_client.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Set up storage context and create index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    shutil.rmtree(temp_dir)
    return index

def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages of a PDF file as images.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def pil_image_to_base64(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 encoded string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def process_question_with_llamaindex(question: str, index) -> str:
    """
    Process a user question using LlamaIndex and return the response.
    """
    memory = ChatMemoryBuffer.from_defaults(token_limit=15000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=("""You have the ability to retrieve information from a database to answer user questions accurately and efficiently.

        The context is as follows:
        ---------------------
        {context_str}
        ---------------------
        Only use the information from the context and do not use external information to answer user questions: {query_str}.

        Remember:
        - Provide complete information, ensuring that it is neither too short nor incomplete.
        - Always use the provided tools to search for information before giving an answer.
        - Respond concisely and clearly.
        - Never fabricate information.
        """
        ),
    )
    response = chat_engine.chat(question)
    return response
