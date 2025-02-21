import os
from typing import Optional, Any
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from rich.console import Console
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback
from rich.pretty import pprint
import json
from pathlib import Path
import logging

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
