import logging
from typing import Dict, Any, List
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from .workflow import FoodOrderingWorkflow # Relative import

logger = logging.getLogger("food_ordering_bot.factory")

def create_chat_engine(menu: Dict[str, Dict[str, Any]], chat_history: List[Dict[str, str]] = None, timeout: float = 60.0):
    """
    Creates the FoodOrderingWorkflow instance.

    Args:
        menu: Dictionary containing menu items.
        chat_history: List of previous chat messages (user/assistant dicts).
        timeout: Workflow execution timeout in seconds.

    Returns:
        A configured FoodOrderingWorkflow instance.
    """
    formatted_chat_history = []
    if chat_history:
        # Keep only the most recent 20 messages (adjust as needed)
        recent_history = chat_history[-20:]
        logger.info(f"Using the last {len(recent_history)} messages out of {len(chat_history)} for history.")
        for message in recent_history:
            role = MessageRole.USER if message["role"] == "user" else MessageRole.ASSISTANT
            formatted_chat_history.append(ChatMessage(role=role, content=message["content"]))

    # Create and return workflow
    workflow = FoodOrderingWorkflow(
        menu=menu,
        chat_history=formatted_chat_history,
        timeout=timeout
    )

    # Note: Workflow steps are registered via decorators in the Workflow class itself.
    logger.info(f"Created FoodOrderingWorkflow with timeout={timeout}")
    return workflow 