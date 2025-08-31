from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

loader = WebBaseLoader("https://www.apple.com/")
docs = loader.load()

print(len(docs))
print("\n\n")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
# print(docs[0].page_content.strip())
response = llm.invoke("Summarize the following document: {}".format(docs[0].page_content.strip()))
print(response.content)
print("\n\n")