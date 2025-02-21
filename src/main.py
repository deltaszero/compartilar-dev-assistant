import operator
import os
import random
import subprocess
import sys
import uuid
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
# initializing the vector store...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PERSIST_DIRECTORY = "/home/dusoudeth/Documentos/github/compartilar-dev-assistant/db/vector_store"
# # remove existing vector store
# if os.path.exists(PERSIST_DIRECTORY):
#     os.system(f"rm -rf {PERSIST_DIRECTORY}")

embeddings = OpenAIEmbeddings(
    # model="text-embedding-ada-002",
    model="text-embedding-3-small",
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
# tools...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@tool
def tokens_from_string(string: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def load_file_contents(file_paths: List[str], vector_store: Chroma, overwrite: bool = False) -> dict:
    logger = RichLogger("utils")
    file_contents = {}
    all_splits = []
    
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
        # query to check if `path` is already in the vector store
        metadata = vector_store.get(
            # limit = 1,
            where = {"source" : {"$eq" : path}}
        )
        existing_metadata = len(metadata["metadatas"]) > 0
        print(metadata["metadatas"])
        if existing_metadata and not overwrite:
            logger.warning(f"File already loaded: {path}")
            continue
        try:
            file_extension = Path(path).suffix.lower()
            documents = []
            # pdf
            if file_extension == '.pdf':
                logger.info(f"Loading PDF file: {path}")
                loader = PyPDFLoader(path)
                documents = loader.load()
            # docx or doc
            elif file_extension in ['.docx', '.doc']:
                logger.info(f"Loading Word document: {path}")
                loader = Docx2txtLoader(path)
                documents = loader.load()
            # other, mainly text files
            else:
                logger.info(f"Loading text file: {path}")
                loader = TextLoader(path)
                documents = loader.load()
            # combining all pages into a single string...
            file_contents[path] = "\n".join(doc.page_content for doc in documents)
            # splitting documents...
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 4098,
                chunk_overlap = 256,
                length_function = len,
                is_separator_regex = False,
            )
            splits = text_splitter.split_documents(documents)
            # addding page numbers and source to metadata...
            for i, split_doc in enumerate(splits):
                original_page = split_doc.metadata.get('page', None)
                if original_page is not None:
                    split_doc.metadata['page_number'] = original_page
                else:
                    split_doc.metadata['page_number'] = i
                split_doc.metadata['source'] = path
            # adding all splits to the list...
            all_splits.extend(splits)
            # calculate tokens per chunk
            tokens_per_chunk = [tokens_from_string(doc.page_content) for doc in splits]
            logger.info(f"Tokens per chunk: {tokens_per_chunk}")
            logger.success(f"Successfully processed: {path}")
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
            continue
    # adding all documents to vector store at once...
    if all_splits:
        vector_store.add_documents(all_splits)
    logger.section("File Loading Summary")
    logger.dict({
        "total_files": len(file_paths),
        "processed_files": len(file_contents),
        "processed_paths": list(file_contents.keys())
    })
    return file_contents


def put_info_into_memory(content:dict, config, *, memory):
    user_id = config["user_id"]
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    memory.put(
        namespace,
        memory_id,
        content
    )
    return {"messages": ["User information saved."]}


def get_info_from_memory(query, config, *, memory):
    user_id = config["user_id"]
    namespace = (user_id, "memories")
    memories = memory.search(
        namespace,
        query
    )
    return {"messages": memories}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining the pipeline state...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class PipelineState(TypedDict):
    messages: List[HumanMessage]
    username: str
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
app = workflow.compile(checkpointer=MemorySaver(), store=in_memory_store)

display_graph(app)


def run_workflow(
        query: dict,
        file_paths: List[str] = list(),
        vector_store: Chroma = vector_store
    ):
    initial_state = {
        "messages": [HumanMessage(content=query)],
    }
    config = {
        "recursion_limit" : 256,
        "configurable" : {
            "thread_id" : "session_test",
            "user_id" : "thiagodsd"
        }
    }
    vector_store = load_file_contents(
        file_paths,
        vector_store,
        False
    )
    for event in app.stream(initial_state, config=config):
        if "__end__" not in event:
            if "process_input" in event:
                print(event["process_input"]["output"])


if __name__ == "__main__":
    query = """
    What is my name?
    """
    run_workflow(
        query,
        [
            "/home/dusoudeth/Documentos/github/compartilar/app/components/friendship/FriendList.tsx",
            "/home/dusoudeth/Documentos/github/compartilar/.firestore-rules",
            "/home/dusoudeth/Documentos/github/compartilar/.storage-rules",
            "/home/dusoudeth/Documentos/github/compartilar/types/friendship.types.ts",
            "/home/dusoudeth/Documentos/github/compartilar/app/components/friendship/FriendSearch.tsx",
            "/home/dusoudeth/Documentos/github/compartilar/app/components/friendship/FriendRequests.tsx",
        ],
        vector_store
    )
