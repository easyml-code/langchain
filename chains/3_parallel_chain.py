from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model1 = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.3
)
model2 = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)
final_chain = parallel_chain | prompt3 | model1 | parser

response = final_chain.invoke({
    "text": """Python is a high-level, interpreted 
    programming language known for its readability and simplicity. 
    It supports multiple programming paradigms, including procedural, 
    object-oriented, and functional programming. Python"""
})
final_chain.get_graph().print_ascii()

print(response)

