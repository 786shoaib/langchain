from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uvicorn
import os
from dotenv import load_dotenv

from langserve import add_routes

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# create prompt template
system_template = "Translate the following into {language}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system",system_template),
    ("user","{text}")
])

# create model
model = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)

# create parser
parser = StrOutputParser()

# create chain
chain = prompt_template | model | parser

app = FastAPI(
    title="My LLM API",
    description="My first LLM API",
    version="1.0"
)

add_routes(
    app,
    chain,
    path = "/chain"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)