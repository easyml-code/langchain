from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()  

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.3
)

result = llm.invoke("What is the capital of France?")
print(result.content)