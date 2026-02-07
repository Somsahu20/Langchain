from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chat_prompt_template import chat_template, prompt

#! chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])


chat_history = []

#! load chat history
with open('history.txt') as f:
    chat_history.extend(f.readlines())

# print(chat_history)

prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my money?'})
print(prompt)
