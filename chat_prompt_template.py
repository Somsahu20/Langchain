from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

chat_template = ChatPromptTemplate([
    # SystemMessage(content=f"You are a helpful {domain} expert"), #type: ignore
    # HumanMessage(content=f"Explain in simple terms, what is {topic}") #type: ignore

    ("system", "You are a helpful {domain} expert"),
    ("human", "Explain in simple terms, what is {topic}?")

])

prompt = chat_template.invoke({"domain": "cricket", "topic": "Doosra"})

print(prompt)