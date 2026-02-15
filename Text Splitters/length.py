from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("books/book1.pdf")
text = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separator="\n\n"
)

texts = splitter.split_documents(text)

print(texts[1].page_content)