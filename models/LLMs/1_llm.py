import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()


client = InferenceClient(
    provider="featherless-ai"
)

# completion = client.chat.completions.create(
#     model="Qwen/Qwen2-7B-Instruct",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is your name?"
#         }
#     ],
# )
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": query
            }
        ],
    )
    print(f"AI: {completion.choices[0].message.content}")
