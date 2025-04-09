import os
import time
from typing import Dict, Any, List, Optional, Union, ClassVar, Type
import logging
from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Context, Event, step, StopEvent, StartEvent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from utils import menu_to_string

# Set up logger
logger = logging.getLogger("food_ordering_bot")

# Define custom event classes
class ResponseEvent(Event):
    """Event containing the final response content."""
    response: str
    action_type: str

class ChatResponseStopEvent(StopEvent):
    """Custom StopEvent with response and action_type fields."""
    response: str
    action_type: str

class FoodOrderingWorkflow(Workflow):
    """
    A workflow that implements the food ordering chatbot conversation logic.
    """
    # Define the start event class
    start_event_cls: ClassVar[Type[Event]] = StartEvent
    
    def __init__(self, menu: Dict[str, Dict[str, Any]], chat_history: List[ChatMessage] = None, timeout: float = 60.0):
        # Configure LLM settings - do this before calling super().__init__()
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.7, request_timeout=30)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Call super().__init__() with explicit timeout
        super().__init__(timeout=timeout)
        
        # Set instance attributes
        self.menu = menu
        self.chat_history = chat_history or []
        self.menu_text = menu_to_string(menu)
        logger.info(f"FoodOrderingWorkflow initialized with timeout={timeout}")
    
    @step
    async def classify_and_respond(self, ctx: Context, ev: StartEvent) -> ResponseEvent:
        """
        First step: Classify intent and generate response
        """
        # Get the query from the start event
        query = ev.content
        logger.info(f"Processing query: '{query}'")
        
        # Determine intent with prompt
        router_prompt = f"""
        Classify user intent as MENU, ORDER, GENERAL, or END.
        User message: "{query}"
        Return only one word from the options.
        """
        
        llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=30)
        logger.info("Sending intent classification request to OpenAI")
        
        # Measure response time for intent classification
        start_time = time.time()
        intent_response = await llm.acomplete(router_prompt)
        elapsed = time.time() - start_time
        
        intent = intent_response.text.strip().lower()
        logger.info(f"Intent classified as: '{intent}' (took {elapsed:.2f}s)")
        
        # Generate appropriate response based on intent
        response = ""
        action_type = ""
        
        try:
            if "menu" in intent:
                logger.info("Handling menu query")
                response = await self._handle_menu_query(query)
                action_type = "menu_inquiry"
            elif "order" in intent:
                logger.info("Handling order query")
                response = await self._handle_order_query(query)
                action_type = "order_action"
            elif "end" in intent:
                logger.info("Handling end conversation")
                response = await self._handle_end_conversation(query)
                action_type = "end_conversation"
            else:  # general conversation
                logger.info("Handling general query")
                response = await self._handle_general_query(query)
                action_type = "general_conversation"
                
            logger.info(f"Generated response (first 50 chars): {response[:50]}...")
            
            # Return a ResponseEvent with the response and action_type
            return ResponseEvent(
                response=response,
                action_type=action_type
            )
        except Exception as e:
            logger.error(f"Error in classify_and_respond: {type(e).__name__}: {str(e)}")
            return ResponseEvent(
                response=f"I'm sorry, I couldn't process your request. Error: {str(e)}",
                action_type="error"
            )
    
    @step
    async def finalize(self, ctx: Context, ev: ResponseEvent) -> ChatResponseStopEvent:
        """
        Final step: Convert ResponseEvent to ChatResponseStopEvent
        """
        logger.info(f"Finalizing response: {ev.response[:30]}...")
        # Create our custom ChatResponseStopEvent with proper fields
        result = ChatResponseStopEvent(
            # Set 'result' to None (required by StopEvent) 
            result=None,
            # Add our custom fields
            response=ev.response,
            action_type=ev.action_type
        )
        logger.info(f"Created ChatResponseStopEvent with fields: response={result.response[:20]}..., action_type={result.action_type}")
        return result
    
    async def _handle_menu_query(self, query: str) -> str:
        """Handle menu-related queries"""
        menu_template = f"""
        You are a helpful assistant providing information about menu items.
        Be friendly and informative about prices and descriptions.
        The complete menu is as follows:
        {self.menu_text}
        """
        
        # Generate response
        messages = self.chat_history + [
            ChatMessage(role=MessageRole.USER, content=query),
            ChatMessage(role=MessageRole.SYSTEM, content=menu_template)
        ]
        
        logger.info("_handle_menu_query: Sending request to OpenAI")
        llm = OpenAI(model="gpt-4o", temperature=0.7, request_timeout=30)
        try:
            # Measure response time for menu query
            start_time = time.time()
            response = await llm.achat(messages)
            elapsed = time.time() - start_time
            
            logger.info(f"_handle_menu_query: Got response in {elapsed:.2f}s")
            return response.message.content
        except Exception as e:
            logger.error(f"_handle_menu_query: Error: {type(e).__name__}: {str(e)}")
            return f"I'm sorry, I had trouble providing menu information. Error: {str(e)}"
    
    async def _handle_order_query(self, query: str) -> str:
        """Handle order-related actions"""
        order_template = f"""
        You are an assistant helping with food orders.
        Based on the menu information and the user's request,
        help them place or modify their order.
        The complete menu is as follows:
        {self.menu_text}

        Be clear about prices and available options.
        If they want to order something not on the menu,
        politely inform them it's not available.
        """
        
        # Generate response
        messages = self.chat_history + [
            ChatMessage(role=MessageRole.USER, content=query),
            ChatMessage(role=MessageRole.SYSTEM, content=order_template)
        ]
        
        logger.info("_handle_order_query: Sending request to OpenAI")
        llm = OpenAI(model="gpt-4o", temperature=0.7, request_timeout=30)
        try:
            # Measure response time for order query
            start_time = time.time()
            response = await llm.achat(messages)
            elapsed = time.time() - start_time
            
            logger.info(f"_handle_order_query: Got response in {elapsed:.2f}s")
            return response.message.content
        except Exception as e:
            logger.error(f"_handle_order_query: Error: {type(e).__name__}: {str(e)}")
            return f"I'm sorry, I had trouble with your order. Error: {str(e)}"
    
    async def _handle_general_query(self, query: str) -> str:
        """Handle general conversation"""
        general_template = """
        You are a friendly restaurant chatbot assistant.
        Respond helpfully to the user's message based on the conversation context.
        Your goal is to be helpful and engage the user.
        """
        
        # Generate response
        messages = self.chat_history + [
            ChatMessage(role=MessageRole.USER, content=query),
            ChatMessage(role=MessageRole.SYSTEM, content=general_template)
        ]
        
        logger.info("_handle_general_query: Sending request to OpenAI")
        llm = OpenAI(model="gpt-4o", temperature=0.7, request_timeout=30)
        try:
            # Measure response time for general query
            start_time = time.time()
            response = await llm.achat(messages)
            elapsed = time.time() - start_time
            
            logger.info(f"_handle_general_query: Got response in {elapsed:.2f}s")
            return response.message.content
        except Exception as e:
            logger.error(f"_handle_general_query: Error: {type(e).__name__}: {str(e)}")
            return f"I'm sorry, I had trouble responding. Error: {str(e)}"
    
    async def _handle_end_conversation(self, query: str) -> str:
        """Handle end of conversation"""
        end_template = """
        The user seems to be ending the conversation.
        Respond with a friendly goodbye message that invites them to return.
        """
        
        messages = self.chat_history + [
            ChatMessage(role=MessageRole.USER, content=query),
            ChatMessage(role=MessageRole.SYSTEM, content=end_template)
        ]
        
        logger.info("_handle_end_conversation: Sending request to OpenAI")
        llm = OpenAI(model="gpt-4o", temperature=0.7, request_timeout=30)
        try:
            # Measure response time for end conversation
            start_time = time.time()
            response = await llm.achat(messages)
            elapsed = time.time() - start_time
            
            logger.info(f"_handle_end_conversation: Got response in {elapsed:.2f}s")
            return response.message.content
        except Exception as e:
            logger.error(f"_handle_end_conversation: Error: {type(e).__name__}: {str(e)}")
            return f"I'm sorry, I had trouble saying goodbye. Error: {str(e)}"

def create_chat_engine(menu: Dict[str, Dict[str, Any]], chat_history: List[Dict[str, str]] = None):
    """
    Create a chat workflow that follows the architectural diagram.
    
    Args:
        menu: Dictionary containing menu items
        chat_history: List of previous chat messages
    
    Returns:
        A workflow object that can be used to process chat messages
    """
    # Convert chat history to the format expected by the workflow
    formatted_chat_history = []
    if chat_history:
        for message in chat_history:
            role = MessageRole.USER if message["role"] == "user" else MessageRole.ASSISTANT
            formatted_chat_history.append(ChatMessage(role=role, content=message["content"]))
    
    # Create and return workflow with an extended timeout (60 seconds)
    return FoodOrderingWorkflow(
        menu=menu, 
        chat_history=formatted_chat_history,
        timeout=60.0  # Increase the timeout to 60 seconds
    ) 