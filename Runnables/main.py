
from abc import ABC, abstractmethod
import random
from re import template

class Runable(ABC):

    @abstractmethod

    def invoke(input_data):
        pass


class FakeLLM(Runable):

    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            "Delhi is the capial of India",
            "AI stands for artificial intelligence",
            "Dhaka is the capital of Bangladesh"
        ]

        return {'response': random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            "Delhi is the capial of India",
            "AI stands for artificial intelligence",
            "Dhaka is the capital of Bangladesh"
        ]

        return {'response': random.choice(response_list)}

class FakePromptTemplate(Runable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

template = FakePromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=['lenght', 'topic']
)

prompt = template.format({'length': 'short', 'topic' : 'India'})

llm = FakeLLM()

# print(llm.predict(prompt))

class FakeChain:

    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predicate(final_prompt)

        return result.response

class RunnableConncector(Runable):

    def __init__(self, runnable_list) -> None:
        self.runnable_list = runnable_list


    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data



chain = RunnableConncector([template, llm])

print(chain.invoke({'length': 'long', 'topic':'India'}))


