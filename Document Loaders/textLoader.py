from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()
loader = TextLoader('football.txt', encoding='utf-8')

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template="Write a summary for the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | llm | parser


docs = loader.load() #! Loads the documnet

# print(docs[0].page_content)

res = chain.invoke({'text': docs[0].page_content})

print(res)