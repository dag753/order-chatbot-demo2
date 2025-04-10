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
import json

# Set up logger
logger = logging.getLogger("food_ordering_bot")

# Define custom event classes
class ResponseEvent(Event):
    """Event containing the response content and action type."""
    response: str
    action_type: str
    original_query: Optional[str] = None # Add field to carry original query

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
        First step: Classify intent. If menu/order, return ack message.
        If greeting/end/irrelevant, return final response directly.
        """
        # Get the query from the start event
        query = ev.content # Correct way to access StartEvent content
        logger.info(f"Processing query: '{query}'")
        
        # Determine intent with prompt - REMOVED menu_text from here
        router_prompt = f"""
        You are a restaurant chatbot assistant. Your primary function is to help users with menu inquiries and placing food orders. Classify the user's intent based on whether it relates to these functions or is a simple greeting/farewell.

        User message: "{query}"

        Classify the intent into one of these categories:
        - MENU: User is asking about menu items, prices, descriptions, or availability (e.g., "What pizzas do you have?", "How much is the burger?").
        - ORDER: User wants to create, modify, review, or cancel an order (e.g., "I want to order food", "Add a pizza to my order", "What's in my cart?", "Can I place an order?").
        - GREETING: User is initiating the conversation (e.g., "hello", "hi", "good morning").
        - END: User is likely ending the conversation (e.g., "bye", "thank you", "that's all").
        - IRRELEVANT: Any other request, instruction, or question not directly related to the menu, ordering, greetings, or farewells.

        If the intent is GREETING, provide a friendly welcome message.
        If the intent is IRRELEVANT, provide a polite refusal message stating you can only assist with menu questions and food orders.

        Return your response STRICTLY as a JSON object with the following structure, ensuring no extra text before or after the JSON:
        {{
            "intent": "MENU|ORDER|GREETING|END|IRRELEVANT",
            "response": "Your response text ONLY if intent is GREETING or IRRELEVANT, otherwise empty string"
        }}
        Example valid JSON output: {{"intent": "ORDER", "response": ""}}
        Another valid example: {{"intent": "IRRELEVANT", "response": "I can only help with menu items and orders."}}
        """
        
        llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=30)
        logger.info("Sending intent classification request to OpenAI")
        
        # Measure response time for intent classification
        start_time = time.time()
        router_response = await llm.acomplete(router_prompt)
        elapsed = time.time() - start_time
        
        # Parse the JSON response
        response_text = router_response.text.strip()
        logger.info(f"Router response (took {elapsed:.2f}s): {response_text}")
        
        # Extract intent and direct response from JSON
        intent = ""
        direct_response = "" # Renamed from general_response for clarity
        
        try:
            # Parse the JSON response
            response_data = json.loads(response_text)
            
            # Extract the fields with better validation
            if isinstance(response_data, dict):
                intent = response_data.get("intent", "").lower() if response_data.get("intent") else ""
                direct_response = response_data.get("response", "") if response_data.get("response") else ""
                
                # Validate the extracted data
                valid_intents = ["menu", "order", "greeting", "end", "irrelevant"]
                if not intent or intent not in valid_intents:
                    # Default to irrelevant if intent is invalid or missing
                    logger.warning(f"Invalid or missing intent '{intent}', defaulting to 'irrelevant'")
                    intent = "irrelevant"
                    # Ensure we provide a default refusal if the intent was bad AND no response was given
                    if not direct_response:
                         direct_response = "I'm sorry, I encountered an issue. I can only assist with menu questions and food orders."

                # If intent is GREETING or IRRELEVANT, check response validity
                if intent in ["greeting", "irrelevant"]:
                   if not isinstance(direct_response, str) or not direct_response.strip():
                       logger.warning(f"Invalid or empty direct response for intent '{intent}': '{direct_response}'. Using fallback message.")
                       # Provide a default response based on the (potentially defaulted) intent
                       if intent == "greeting":
                            direct_response = "Hello! How can I assist you with the menu or an order?"
                       else: # irrelevant
                            direct_response = "I'm sorry, I can only assist with questions about our menu and help you place an order."
                   elif direct_response.strip() in ['{', '}', '[]', '[', ']', '{}', ':', '""', "''", ',', '.']:
                       logger.warning(f"Direct response for intent '{intent}' looks like a fragment: '{direct_response}'. Using fallback message.")
                       if intent == "greeting":
                            direct_response = "Hello! How can I assist you with the menu or an order?"
                       else: # irrelevant
                            direct_response = "I'm sorry, I can only assist with questions about our menu and help you place an order."
                
                logger.info(f"Successfully parsed JSON response: intent='{intent}', response_length={len(direct_response)}")
            else:
                logger.error("Response data is not a dictionary, defaulting to irrelevant")
                intent = "irrelevant"
                direct_response = "I'm sorry, I encountered an issue processing the response. I can only assist with menu questions and food orders."
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            logger.error(f"Failed to parse JSON response: {e}")
            logger.info(f"Raw response text was: {response_text}")
            logger.info("Attempting keyword-based intent extraction as fallback...")
            
            # Simple keyword check on the raw text
            raw_text_lower = response_text.lower()
            if "menu" in raw_text_lower:
                intent = "menu"
                logger.info("Fallback: Detected 'menu' keyword.")
            elif "order" in raw_text_lower or "cart" in raw_text_lower or "checkout" in raw_text_lower:
                intent = "order"
                logger.info("Fallback: Detected 'order/cart/checkout' keyword.")
            elif any(greeting in raw_text_lower for greeting in ["hello", "hi ", " how are"]):
                 intent = "greeting"
                 direct_response = "Hello! How can I help you with the menu or your order today?" # Provide default greeting
                 logger.info("Fallback: Detected greeting keyword.")
            elif any(farewell in raw_text_lower for farewell in ["bye", "thank you", "thanks"]):
                 intent = "end"
                 logger.info("Fallback: Detected farewell keyword.")
            else:
                # Only default to irrelevant if no keywords match
                intent = "irrelevant"
                direct_response = "I'm sorry, I had trouble understanding that. I can only assist with menu questions and food orders." # Specific message for this fallback path
                logger.info("Fallback: No relevant keywords detected, defaulting to irrelevant.")
                
        logger.info(f"Intent classified as: '{intent}' (took {elapsed:.2f}s)")
        
        # Generate appropriate response or acknowledgment based on intent
        response = ""
        action_type = ""
        final_response_event = None

        try:
            if intent == "menu":
                logger.info("Intent: MENU. Returning acknowledgment.")
                response = "Give us a moment while we research that for you."
                action_type = "menu_inquiry_pending" # Temporary type
                final_response_event = ResponseEvent(
                    response=response,
                    action_type=action_type,
                    original_query=query # Pass query for next step
                )
            elif intent == "order":
                logger.info("Intent: ORDER. Returning acknowledgment.")
                # Basic check for modification keywords - can be enhanced
                if any(kw in query.lower() for kw in ["change", "modify", "add", "remove", "update"]):
                     response = "Give us a moment while we get that order modification ready for you."
                else:
                     response = "Give us a moment while we get that order ready for you."
                action_type = "order_action_pending" # Temporary type
                final_response_event = ResponseEvent(
                    response=response,
                    action_type=action_type,
                    original_query=query # Pass query for next step
                )
            elif intent == "greeting":
                logger.info("Handling greeting directly from router")
                response = direct_response
                action_type = "greeting"
                final_response_event = ResponseEvent(response=response, action_type=action_type)
            elif intent == "end":
                logger.info("Handling end conversation")
                # Call _handle_end_conversation directly as it's simple
                response = await self._handle_end_conversation(query)
                action_type = "end_conversation"
                final_response_event = ResponseEvent(response=response, action_type=action_type)
            elif intent == "irrelevant":
                 logger.info("Handling irrelevant query directly from router")
                 response = direct_response
                 action_type = "irrelevant_query"
                 final_response_event = ResponseEvent(response=response, action_type=action_type)
            else: # Should not happen due to validation, but good to have a fallback
                logger.error(f"Reached unexpected else block for intent: {intent}")
                response = "I'm sorry, I'm not sure how to handle that. I can assist with menu questions and orders."
                action_type = "error"
                final_response_event = ResponseEvent(response=response, action_type=action_type)

            logger.info(f"Step 1 Result: Type='{action_type}', Response='{response[:50]}...'")
            return final_response_event
            
        except Exception as e:
            logger.error(f"Error in classify_and_respond logic block: {type(e).__name__}: {str(e)}")
            # Provide a generic refusal in case of errors in handlers
            return ResponseEvent(
                response="I'm sorry, I encountered an error and cannot process your request. I can only assist with menu questions and food orders.",
                action_type="error"
            )
    
    @step
    async def generate_detailed_response(self, ctx: Context, ev: ResponseEvent) -> ResponseEvent:
        """
        Second step: If the previous step returned a pending action type,
        generate the detailed response using the appropriate handler.
        Otherwise, pass the event through.
        """
        logger.info(f"Entering generate_detailed_response with action_type: {ev.action_type}")
        
        if ev.action_type == "menu_inquiry_pending":
            logger.info("Handling pending menu inquiry")
            if ev.original_query:
                response_text = await self._handle_menu_query(ev.original_query)
                logger.info(f"Generated detailed menu response: {response_text[:50]}...")
                return ResponseEvent(
                    response=response_text,
                    action_type="menu_inquiry" # Final action type
                    # original_query is no longer needed
                )
            else:
                logger.error("Original query missing for menu_inquiry_pending")
                return ResponseEvent(response="Error: Missing query for menu info.", action_type="error")
                
        elif ev.action_type == "order_action_pending":
            logger.info("Handling pending order action")
            if ev.original_query:
                response_text = await self._handle_order_query(ev.original_query)
                logger.info(f"Generated detailed order response: {response_text[:50]}...")
                return ResponseEvent(
                    response=response_text,
                    action_type="order_action" # Final action type
                )
            else:
                 logger.error("Original query missing for order_action_pending")
                 return ResponseEvent(response="Error: Missing query for order action.", action_type="error")
                 
        else:
            # If action type is already final (greeting, end, irrelevant, error), pass through
            logger.info(f"Passing through response with final action_type: {ev.action_type}")
            return ev

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
        """Handle general conversation (DEPRECATED - Now handled by GREETING/IRRELEVANT in classify_and_respond)"""
        # This method is no longer called by the main logic but kept for potential future use or reference.
        logger.warning("_handle_general_query called, but this path should be deprecated.")
        return "I'm sorry, I can only assist with questions about our menu and help you place an order."
    
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
    workflow = FoodOrderingWorkflow(
        menu=menu, 
        chat_history=formatted_chat_history,
        timeout=60.0  # Increase the timeout to 60 seconds
    )
    
    # The @step decorators automatically register and link the steps based on type hints.
    # Explicit add_step calls are not needed here.
    # workflow.add_step(workflow.classify_and_respond)
    # workflow.add_step(workflow.generate_detailed_response, input_step_name="classify_and_respond")
    # workflow.add_step(workflow.finalize, input_step_name="generate_detailed_response")
    
    return workflow 