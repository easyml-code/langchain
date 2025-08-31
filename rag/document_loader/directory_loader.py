from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

loader = DirectoryLoader(path = "rag/document_loader/data/directory/", 
                         glob="**/*.pdf",
                         loader_cls=PyPDFLoader)

documents = loader.load()

print(f"Loaded {len(documents)} documents")
print("\n\n")
for doc in documents:
    print(f"Document loaded name: {doc.metadata['source']}, page: {doc.metadata['page']+1}")
    print("\n---\n")
print("\n\n")
print(documents[0].metadata)