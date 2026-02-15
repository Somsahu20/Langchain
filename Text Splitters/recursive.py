from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("books/book1.pdf")
text = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

chunks = splitter.split_documents(text)

for chunk in chunks:
    print(f"-----------------------------------------\n{chunk}\n---------------------------------")
