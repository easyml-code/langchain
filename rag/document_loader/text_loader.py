from langchain_community.document_loaders import TextLoader
loader = TextLoader("/Users/vikram/Documents/projects/langchain/rag/document_loader/data/sample.txt", encoding="utf-8")

docs = loader.load()
print(len(docs))
print(docs[0])