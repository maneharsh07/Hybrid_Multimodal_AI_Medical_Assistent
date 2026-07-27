#-------------------------------------------CREATE FAISS VECTOR DB---------------------------------------------

import os
from dotenv import load_dotenv

# Load .env file properly
load_dotenv(dotenv_path=".env")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


#-------------------------------------------CHECK API KEY---------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please check your .env file.")


#-------------------------------------------LOAD DATA---------------------------------------------

file_path = "medical_data.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ {file_path} not found. Please create it.")

loader = TextLoader(file_path)
documents = loader.load()


#-------------------------------------------TEXT SPLITTING---------------------------------------------

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)


#-------------------------------------------EMBEDDINGS---------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)


#-------------------------------------------CREATE & SAVE FAISS---------------------------------------------

db = FAISS.from_documents(texts, embeddings)

save_path = "Medical_chatbot/vectorstore/db_faiss"

os.makedirs(save_path, exist_ok=True)

db.save_local(save_path)


#-------------------------------------------DONE---------------------------------------------

print("✅ FAISS DB Created Successfully!")
print(f"📁 Saved at: {save_path}")