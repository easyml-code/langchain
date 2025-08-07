from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()   

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=512
)

parser = StrOutputParser()

chain = prompt | model | parser


from langchain_core.messages import AIMessage, AIMessageChunk
from typing import Iterable
def khud_ka_parser(input: AIMessage) -> str:
    return input.content[::-1] # reverse the string

# chain2 = prompt | model | khud_ka_parser
# result = chain2.invoke({"topic": "5 word intro"})
# print(result)

# import time
# import random
# for chunk in chain.stream({"topic": "Python programming language"}):
#     print(chunk, end="", flush=True)
#     time.sleep(random.uniform(0.05, 0.15))



# response = chain.invoke({"topic": "Python programming language"})
# print(response)


from langchain_core.runnables import RunnableGenerator


def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
    for chunk in chunks:
        yield chunk.content.swapcase()


streaming_parse = RunnableGenerator(streaming_parse)

chain3 = prompt | model | streaming_parse
result = chain3.invoke({"topic": "Python programming language"})
print(result)