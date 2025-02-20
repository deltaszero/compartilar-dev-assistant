import os
import asyncio
from typing import Annotated, List, Tuple, TypedDict, Optional, Any
import operator
from datetime import datetime  # noqa: F401
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, BaseMessage  # noqa: F401
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore  # noqa: F401
from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback
from rich.pretty import pprint
import json
from pathlib import Path
import logging

load_dotenv("../credentials.env")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# utility functions...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "highlight": "magenta"
})

class RichLogger:
    def __init__(self, name: str = "app", log_dir: str = "logs"):
        # Initialize rich console
        self.console = Console(theme=theme)
        install_rich_traceback()
        
        # Set up log directory
        self.log_path = Path(log_dir)
        self.log_path.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Create file handler with timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_path / f"{name}.log"
        
        # Add file handler with formatter
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Keep track of the current log file
        self.current_log_file = log_file
        
        self.info(f"Logging initialized. Log file: {log_file}")

    def _format_for_file(self, data: Any) -> str:
        """Format data for file logging"""
        if isinstance(data, dict):
            return json.dumps(data, indent=2, default=str)
        return str(data)

    def print_json(self, data: Any, title: Optional[str] = None):
        """Pretty print JSON-serializable data"""
        if title:
            self.console.print(f"\n[bold]{title}[/bold]")
            self.console.print("─" * 40)
            self.logger.info(f"=== {title} ===")
        
        try:
            # Console output
            json_str = json.dumps(data, indent=2, default=str)
            parsed = json.loads(json_str)
            pprint(parsed, expand_all=True)
            
            # File output
            self.logger.info(f"\n{json_str}")
        except Exception as e:
            error_msg = f"Error formatting data: {str(e)}"
            self.console.print(f"[error]{error_msg}[/error]")
            self.logger.error(error_msg)
            # Fallback to string representation for file
            self.logger.info(str(data))

    def info(self, message: str):
        """Log info message"""
        self.console.print(f"[info]ℹ {message}[/info]")
        self.logger.info(message)

    def success(self, message: str):
        """Log success message"""
        self.console.print(f"[success]✓ {message}[/success]")
        self.logger.info(f"SUCCESS: {message}")

    def warning(self, message: str):
        """Log warning message"""
        self.console.print(f"[warning]⚠ {message}[/warning]")
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.console.print(f"[error]✗ {message}[/error]")
        self.logger.error(message)

    def highlight(self, message: str):
        """Print highlighted message"""
        self.console.print(f"[highlight]{message}[/highlight]")
        self.logger.info(f"HIGHLIGHT: {message}")

    def section(self, title: str):
        """Print section header"""
        self.console.rule(f"[bold]{title}")
        self.logger.info(f"\n{'='*50}\n{title}\n{'='*50}")

    def dict(self, data: dict, title: Optional[str] = None):
        """Pretty print dictionary data"""
        if title:
            self.console.print(f"\n[bold]{title}[/bold]")
            self.console.print("─" * 40)
            self.logger.info(f"\n=== {title} ===")
        
        # Console output
        pprint(data, expand_all=True)
        
        # File output - format dictionary as JSON string
        formatted_data = self._format_for_file(data)
        self.logger.info(f"\n{formatted_data}")

    def get_log_file(self) -> Path:
        """Get the current log file path"""
        return self.current_log_file


def load_file_contents(file_paths: List[str]) -> dict:
    """
    Load and index file contents from given paths
    Supports PDF, DOCX, and text files
    """
    file_contents = {}
    
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        try:
            file_extension = Path(path).suffix.lower()
            
            if file_extension == '.pdf':
                logger.info(f"Loading PDF file: {path}")
                loader = PyPDFLoader(path)
                documents = loader.load()
                # Store raw content by joining all pages
                file_contents[path] = "\n".join(doc.page_content for doc in documents)
                
            elif file_extension in ['.docx', '.doc']:
                logger.info(f"Loading Word document: {path}")
                loader = Docx2txtLoader(path)
                documents = loader.load()
                file_contents[path] = "\n".join(doc.page_content for doc in documents)
                
            else:  # Default to text file handling
                logger.info(f"Loading text file: {path}")
                loader = TextLoader(path)
                documents = loader.load()
                # Store raw content
                with open(path, 'r') as f:
                    file_contents[path] = f.read()
            
            # Split content for vectorization regardless of file type
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Add to vector store with metadata
            vector_store.add_documents(splits)
            
            logger.success(f"Successfully processed: {path}")
            
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
            continue
    
    # Log summary
    logger.section("File Loading Summary")
    logger.dict({
        "total_files": len(file_paths),
        "processed_files": len(file_contents),
        "processed_paths": list(file_contents.keys())
    })
    
    return file_contents


def display_graph(graph, output_folder="output", file_name="graph"):
    """
    display graph
    """
    mermaid_png = graph.get_graph(xray=1).draw_mermaid_png(
        draw_method = MermaidDrawMethod.API, 
        curve_style = CurveStyle.NATURAL
    )
    #
    output_folder = "."
    os.makedirs(output_folder, exist_ok=True)
    #
    filename = os.path.join(output_folder, "graph.png")
    with open(filename, 'wb') as f:
        f.write(mermaid_png)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# initializing the vector store...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

logger = RichLogger("pipeline")

PERSIST_DIRECTORY = "../db/vector_store"
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    api_key = os.getenv("OPENAI_API_KEY")
)
vector_store = Chroma(
    persist_directory = PERSIST_DIRECTORY, 
    embedding_function = embeddings
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# initializing the language model and agents...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
llm_gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant."""),
        ("placeholder", "{messages}")
    ]
)

agent_planner = create_react_agent(
    model = llm_gpt_4o_mini,
    tools = list(),
    state_modifier = prompt
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining the pipeline state...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class PipelineState(TypedDict):
    messages: str
    current_plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    active_agent: str
    file_contents: dict
    planner_feedback: str

class Plan(BaseModel):
    steps: List[str] = Field(description="Numbered unique steps to complete the development task, in order")

class Feedback(BaseModel):
    message: str = Field(description="Feedback message from the planner")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# setting up the planning...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PLANNER_PROMPT = """You are a Next.js 14 and Firebase development planner. Plan the development steps.
Task: {input}
Related files: {file_paths}

Create a detailed 3-step plan considering:
1. Next.js 14 best practices and conventions
2. Firebase/Firestore integration requirements
3. TypeScript type safety
4. Security considerations

Provide a step-by-step plan.

You've already completed the following steps:
{past_steps}
"""

# CODE_AGENT_PROMPT = """You are a Next.js 14 and TypeScript expert. Write high-quality code.
# Current step: {current_step}
# Related file contents: {relevant_content}

# Write code following:
# 1. Next.js 14 conventions (app router, server components, etc.)
# 2. TypeScript best practices
# 3. React patterns and hooks
# 4. Clean code principles

# Planner feedback: {planner_feedback}
# """

async def plan_step(state: PipelineState):
    """Create development plan"""
    logger.section("reasoning")
    logger.dict(state, "state")
    #
    if (len(state["current_plan"]) == 0) and (len(state["past_steps"]) == 0):
        planning_chain = ChatPromptTemplate.from_template(PLANNER_PROMPT) | llm_gpt_4o_mini.with_structured_output(Plan)
        plan = await planning_chain.ainvoke({
            "input": state["messages"][-1].content,
            "file_paths": list(state["file_contents"].keys()),
            "past_steps": state["past_steps"]
        })
        return {
            "current_plan": plan.steps,
            "active_agent": "code_agent"
        }
    elif len(state["past_steps"]) > 0:
        planning_chain = ChatPromptTemplate.from_template(PLANNER_PROMPT) | llm_gpt_4o_mini.with_structured_output(Feedback)
        plan = await planning_chain.ainvoke({
            "input": f"""Evaluate the coherence between the user's query and the already completed steps.\n\nUSER QUERY: {state["messages"][-1].content}""",
            "file_paths": list(state["file_contents"].keys()),
            "past_steps": state["past_steps"]
        })
        return {
            "planner_feedback": plan.message,
            "active_agent": "code_agent"
        }


def should_end(state: PipelineState):
    if (len(state["current_plan"]) == 0) and (len(state["past_steps"]) > 0): 
        return END
    else:
        return "code_agent"


async def code_agent_step(state: PipelineState):
    """Generate Next.js/TypeScript code"""
    logger.section("acting")
    logger.dict(state, "state")
    #
    current_step = state["current_plan"]
    current_step_str = "\n".join(
        [f"{i+1}. {step}" for i, step in enumerate(state["current_plan"])]
    )
    # rag
    relevant_docs = vector_store.similarity_search(current_step_str, k=1)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    task = current_step[0]
    task_str = f"""
    You are a Next.js 14 and TypeScript expert. Write high-quality code.

    For the following plan: {current_step_str} You are currently working on: {task}

    Below there is some context from the vector store that might help you to write the code:
    {context}

    Below there is the feedback from the planner: {state["planner_feedback"]}
    """
    logger.info(task_str)
    #
    agent_response = await agent_planner.ainvoke({
        "messages": [("user", task_str)]
    })
    return {
        "past_steps" : [(task, agent_response["messages"][-1].content)],
        "active_agent" : "planner",
        "current_plan": state["current_plan"][1:]
    }

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# defining the workflow...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
workflow = StateGraph(PipelineState)
# adding nodes... 
workflow.add_node("planner", plan_step)
workflow.add_node("code_agent", code_agent_step)
# adding edges...
workflow.add_edge(START, "planner")
workflow.add_edge("code_agent", "planner")
# adding conditional edges...
workflow.add_conditional_edges("planner", should_end, ["code_agent", END])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# compiling the workflow and displaying the graph...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
app = workflow.compile()

display_graph(app)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# setting up the pipeline...
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
async def run_pipeline(query: str, file_paths: List[str]):
    """
    Run the development pipeline
    """
    # loading file contents...
    print(f"loading file contents from: {file_paths}")
    file_contents = load_file_contents(file_paths)
    # defining the initial state...
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_plan": list(),
        "execution_history": list(),
        "active_agent": "planner",
        "file_contents": file_contents,
        "planner_feedback" : ""
    }
    print("running pipeline...")
    async for event in app.astream(initial_state):
        if "__end__" not in event:
            if "messages" in event:
                print(f"Response: {event['messages'][-1].content}")
            if "current_plan" in event:
                print(f"Plan: {event['current_plan']}")


if __name__ == "__main__":
    query = """
    Does the code below have any issues?
    ```tsx
    // app/settings/page.tsx
    'use client';

    import React from 'react';

    export default function SettingsPage() {

        return (
            <div className="h-screen flex flex-col">
            </div>
        );
    }
    ```
    """
    file_paths = [
        "/home/dusoudeth/Documentos/github/compartilar/.storage-rules"
    ]
    asyncio.run(run_pipeline(query, file_paths))