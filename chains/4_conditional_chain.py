from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(
        description="The sentiment of the text"
    )

parser = StrOutputParser()
parser1 = PydanticOutputParser(pydantic_object=Sentiment)
format_instruction = parser1.get_format_instructions()

model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

prompt1 = PromptTemplate(
    template='Analyze the sentiment: {feedback} in {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':format_instruction}
)
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

chain1 = prompt1 | model | parser1

branch = RunnableBranch(
        (lambda x:x.sentiment=="positive", prompt2 | model | parser),
        (lambda x:x.sentiment=="negative", prompt3 | model | parser),
        RunnableLambda(lambda x: "Thank you for your feedback!")
    )

main_chain = chain1 | branch

response = main_chain.invoke({
    "feedback": "I love programming in Python! It's so much fun and rewarding."
})
branch.get_graph().print_ascii()
print(response)
