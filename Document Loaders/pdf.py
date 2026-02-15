from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
loader = PyPDFLoader('frenchrevolution.pdf') #? Only suitable when the pdf consists only of textual data

docs = loader.load()

sz = len(docs)

for i in range(sz):
    print(docs[i].page_content)