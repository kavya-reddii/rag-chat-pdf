import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
hf_api_key = os.getenv('HF_API_KEY')  # Make sure the key is set in .env file
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Login with your Hugging Face token
login(token=hf_api_key)

# Now you can load models from the Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


if not hf_api_key:
    raise ValueError("OpenAI API key is missing. Please add it to your .env file.")

# Set the OpenAI API key as an environment variable
os.environ["HF_API_KEY"] = hf_api_key

def create_vector_store(pdf_folder="pdf_files", index_folder="faiss_index"):
    # Step 1: Read text from PDF files
    texts = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)

    # Step 2: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(texts)

    # Step 3: Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 4: Create FAISS index and save it
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_folder)
    
    return vector_store
