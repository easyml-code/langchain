from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="/Users/vikram/Documents/projects/langchain/rag/document_loader/data/sample_pdf.pdf")

docs = loader.load()

print(len(docs))
print(docs[0])