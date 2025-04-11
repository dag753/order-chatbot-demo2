from typing import Dict, Any, List, Optional
from llama_index.core.workflow import Event, StopEvent

# Define custom event classes
class ResponseEvent(Event):
    """Event containing the response content and action type."""
    response: str
    action_type: str
    original_query: Optional[str] = None # Add field to carry original query
    cart_items: Optional[List[Dict[str, Any]]] = None # Add field for cart items
    cart_status: Optional[str] = None # Add field for cart status
    prompt_tokens: Optional[int] = None # Add field for prompt tokens
    completion_tokens: Optional[int] = None # Add field for completion tokens

class ChatResponseStopEvent(StopEvent):
    """Custom StopEvent with response and action_type fields."""
    response: str
    action_type: str
    cart_items: Optional[List[Dict[str, Any]]] = None # Add field for cart items
    cart_status: Optional[str] = None # Add field for cart status
    prompt_tokens: Optional[int] = None # Add field for prompt tokens
    completion_tokens: Optional[int] = None # Add field for completion tokens 