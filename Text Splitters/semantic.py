from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv




load_dotenv()


text = """

The old farmer woke up at dawn to check his corn crops, noticing that the soil was perfectly moist for the spring planting season. Suddenly, a group of cyber-terrorists bypassed the national firewall and began a massive coordinated attack on the city's power grid to demand a billion dollars in cryptocurrency. As the sirens wailed across the darkened skyline, the farmer decided that he really preferred sourdough bread over wheat because it made much better toast for his morning breakfast.

Knights in heavy iron armor charged across the muddy battlefield, their swords clashing as they fought a desperate war to defend the medieval kingdom from invaders. Just then, a lead surgeon in a high-tech hospital shouted for a nurse to hand him a scalpel because the patient's blood pressure was dropping rapidly during the heart transplant. "He's at the thirty-yard line, he's at the twenty, he's going to go all the way for a touchdown!" the sports announcer screamed as the stadium crowd erupted in a deafening roar of excitement.

Little Sarah laughed with delight as she blew out the six candles on her bright pink birthday cake while her friends sang a happy song. Thousands of feet below the ocean surface, a deep-sea submersible's hull began to creak under the immense pressure of the midnight zone while scientists searched for giant squids. The CEO adjusted his silk tie and pointed at the quarterly earnings graph, reminding the board of directors that the merger would likely lead to a fifteen percent increase in shareholder dividends by next fiscal year.

"""

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.8
)

res = splitter.create_documents([text])


for chunk in res:
    print(chunk)

