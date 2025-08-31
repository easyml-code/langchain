from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='/Users/vikram/Documents/projects/langchain/rag/document_loader/data/customers-100.csv', encoding='utf-8')

docs =loader.load()

print(len(docs))
print(docs[0])