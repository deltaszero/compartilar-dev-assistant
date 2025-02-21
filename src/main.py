import operator
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Annotated, List, Tuple, TypedDict

import openai
import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatMaritalk
from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader,
                                                  TextLoader)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from utils import RichLogger, display_graph

load_dotenv("../credentials.env")

logger = RichLogger("pipeline")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# tools...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@tool
def tokens_from_string(string: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# initializing the vector store...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PERSIST_DIRECTORY = "/home/dusoudeth/Documentos/github/compartilar-dev-assistant/db/vector_store"
# remove existing vector store
if os.path.exists(PERSIST_DIRECTORY):
    os.system(f"rm -rf {PERSIST_DIRECTORY}")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# memories...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
in_memory_store = InMemoryStore()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining the pipeline state...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class PipelineState(TypedDict):
    messages: List[HumanMessage]
    output: str

class TextualResponse(BaseModel):
    output: str = Field(description="textual output of the language model")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# initializing the language model and agents...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."), ("placeholder", "{messages}")
])

llm_gpt_4o_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_sabia_3 = ChatMaritalk(
    model="sabia-3",
    api_key=os.getenv("MARITACA_KEY"),
    temperature=0.01
)

llm_deepseek_v3 = ChatDeepSeek(
    model = "deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.11
)

# create agents...
agent_summarizer = create_react_agent(
    model=llm_deepseek_v3,
    tools=list(),
    state_modifier=prompt
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining nodes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
INPUT_PROCESSING_PROMPT = """
You are a helpful assistant. You are here to help the user with their questions. Please an answer the following question:
{messages}
"""
def process_input(state: PipelineState):
    messages = state["messages"]
    processing_chain = ChatPromptTemplate.from_template(INPUT_PROCESSING_PROMPT) | llm_deepseek_v3.with_structured_output(TextualResponse)
    response = processing_chain.invoke(messages)
    return {"output": response.output}


workflow = StateGraph(MessagesState)
# adding nodes... 
workflow.add_node("process_input", process_input)
# adding edges...
workflow.add_edge(START, "process_input")
workflow.add_edge("process_input", END)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# compiling the workflow and displaying the graph...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
app = workflow.compile()

display_graph(app)


def run_workflow(query: dict):
    initial_state = {
        "messages": [HumanMessage(content=query)],
    }    
    for event in app.stream(initial_state, config={"recursion_limit": 250}):
        if "__end__" not in event:
            if "process_input" in event:
                print(event["process_input"]["output"])


if __name__ == "__main__":
    query = """
    What is the best way to install a new package in Python?
    """
    run_workflow(query)
