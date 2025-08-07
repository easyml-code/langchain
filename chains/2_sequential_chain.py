from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()   

model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

prompt1 = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Summarize the following facts: {facts}',
    input_variables=['facts']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({"topic": "Python programming language"})
print(response)
