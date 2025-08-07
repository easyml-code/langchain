import os
from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

print("Loading HuggingFace model...\n",os.getenv("HUGGINGFACEHUB_API_TOKEN"))

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    provider="featherless-ai"
)

model = ChatHuggingFace(llm=hf_endpoint)

result = model.invoke("What is the capital of India?")
print(result.content)
