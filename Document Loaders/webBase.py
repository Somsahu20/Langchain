from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

url = ['https://en.wikipedia.org/wiki/Franz_Kafka', 
       'https://en.wikipedia.org/wiki/Sigmund_Freud']
loader = WebBaseLoader(url)
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following question \n {questions} from the text \n {text}',
    input_variables=['questions', 'text']
)

docs = loader.load()

chain = prompt | llm | parser

questions = """

1) Who is sigmund freud?
2) Why is sigmund freud considered as the father of psychology?

"""

result = chain.invoke({'questions': questions, 'text': docs[1].page_content})

print(result)