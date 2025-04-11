import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union, ClassVar, Type

from llama_index.core import Settings
from llama_index.core.workflow import Workflow, Context, Event, step, StopEvent, StartEvent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.llms.types import MessageRole, ChatMessage

from utils import menu_to_string
from .events import ResponseEvent, ChatResponseStopEvent
from .handlers import (
    format_response_text,
    handle_menu_query,
    handle_order_query,
    handle_order_confirmation,
    handle_end_conversation,
    _extract_token_counts
)

logger = logging.getLogger("food_ordering_bot.workflow")

class FoodOrderingWorkflow(Workflow):
    """
    A workflow that implements the food ordering chatbot conversation logic.
    Uses separate handlers for different intents.
    """
    start_event_cls: ClassVar[Type[Event]] = StartEvent

    def __init__(self, menu: Dict[str, Dict[str, Any]], chat_history: List[ChatMessage] = None, timeout: float = 60.0):
        if not Settings.llm:
            Settings.llm = OpenAI(model="gpt-4o", request_timeout=30)
        if not Settings.embed_model:
             Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        super().__init__(timeout=timeout)
        self.menu = menu
        self.chat_history = chat_history or []
        self.menu_text = menu_to_string(menu)
        logger.info(f"FoodOrderingWorkflow initialized with timeout={timeout}")

    @step
    async def classify_and_respond(self, ctx: Context, ev: StartEvent) -> Union[ResponseEvent, ChatResponseStopEvent]:
        """
        First step: Classify intent using an LLM.
        If menu/order/confirmation, return ResponseEvent with a pending action type.
        If greeting/end/irrelevant/history, handle directly and return ChatResponseStopEvent.
        """
        query = ev.content
        logger.info(f"Processing query: '{query}'")

        formatted_history = "\n".join([
            f"{'USER' if msg.role == MessageRole.USER else 'ASSISTANT'}: {msg.content}"
            for msg in self.chat_history
        ]) if self.chat_history else "No conversation history yet."

        # --- Intent Classification LLM Call ---
        router_prompt = f"""
        You are a restaurant chatbot assistant classifying user intent.

        **ABSOLUTE RULES - APPLY THESE FIRST:**
        1.  If the user message is a simple, standalone greeting (e.g., "hello", "hi"), the intent is GREETING.
        2.  If the user message is asking *about the conversation itself* (e.g., "what did I ask?", "what was my last message?", "what did we talk about?"), the intent is HISTORY, regardless of the topic of previous messages.
        3.  If the user indicates they are done ordering with phrases like "nothing else", "that's all", "I'm done", "that's it", or similar, the intent is ORDER_CONFIRMATION (not END).

        **Conversation History:**
        {formatted_history}

        **Current User Message:** "{query}"

        Given the user message, conversation history, and the absolute rules, classify the intent into ONE of the following:
        - GREETING: A simple greeting. **Must follow Absolute Rule 1.**
        - HISTORY: A question *about* the conversation history or previous messages/orders. **Must follow Absolute Rule 2.**
        - MENU: Asking about menu items/prices/descriptions. *Excludes questions about what was previously discussed.*
        - ORDER: Requesting to create, modify, review, or cancel an order. *Excludes questions about previous orders already discussed.*
        - ORDER_CONFIRMATION: Confirming a pending order (e.g., "confirm my order", "yes I want to order", "place my order", "I confirm", "that's correct") or indicating they are done ordering (e.g., "nothing else", "that's all", "I'm done"). **Must follow Absolute Rule 3.**
        - END: Ending the conversation (e.g., "bye", "thank you") without intent to complete an order. This is ONLY for final goodbyes, not for completing an order.
        - IRRELEVANT: Any other topic not covered above.

        Output Instructions:
        1. Return STRICTLY a JSON object with keys "intent" and "response".
        2. For GREETING or HISTORY intents: Set the correct "intent". Provide a **concise**, **plain text** response (under 320 characters, no markdown) in the "response" field. Use the provided Conversation History context to answer HISTORY questions accurately. Summarize if necessary and offer to provide more detail if relevant.
        3. For IRRELEVANT intent: Set "intent" to "IRRELEVANT". Provide a **concise**, empathetic **plain text** refusal/explanation (under 320 characters, no markdown) in the "response" field.
        4. For MENU, ORDER, ORDER_CONFIRMATION, or END intents: Set the correct "intent" and set "response" to an empty string ("").

        Examples (History examples assume relevant context was in the provided history):
        User message: "hello"
        Output: {{"intent": "GREETING", "response": "Hello! How can I help with the menu or your order?"}} # Plain text

        User message: "What drinks do you have?"
        Output: {{"intent": "MENU", "response": ""}}

        User message: "I'd like to confirm my order"
        Output: {{"intent": "ORDER_CONFIRMATION", "response": ""}}

        User message: "Yes I want to place this order"
        Output: {{"intent": "ORDER_CONFIRMATION", "response": ""}}

        User message: "nothing else"
        Output: {{"intent": "ORDER_CONFIRMATION", "response": ""}}

        User message: "that's all"
        Output: {{"intent": "ORDER_CONFIRMATION", "response": ""}}

        User message: "I'm done"
        Output: {{"intent": "ORDER_CONFIRMATION", "response": ""}}

        User message: "what did I ask before this?"
        Output: {{"intent": "HISTORY", "response": "You previously asked about our sandwich options. Need more details on those, or can I help with something else?"}} # Plain text

        User message: "What was my previous message?"
        Output: {{"intent": "HISTORY", "response": "Your previous message was asking about drinks. Anything else I can help with?"}} # Plain text

        User message: "what was the first thing I asked?"
        Output: {{"intent": "HISTORY", "response": "Looks like your first message was 'Hello'. How can I help now?"}} # Plain text

        User message: "what did I order last time?"
        Output: {{"intent": "HISTORY", "response": "We discussed you ordering pizza previously. Want to order that now or see the menu again?"}} # Plain text

        User message: "tell me a joke"
        Output: {{"intent": "IRRELEVANT", "response": "Sorry, I can't tell jokes! I'm here for menu questions or orders. Can I help with that?"}} # Plain text

        User message: "thanks bye"
        Output: {{"intent": "END", "response": ""}}

        Ensure no extra text before or after the JSON object.
        """

        llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=30)
        logger.info("Sending intent classification request to OpenAI")
        start_time = time.time()
        router_response = await llm.acomplete(router_prompt)
        elapsed = time.time() - start_time

        prompt_tokens, completion_tokens = _extract_token_counts(router_response, "classify_intent")

        response_text = router_response.text.strip()
        logger.info(f"Router response (took {elapsed:.2f}s, p={prompt_tokens}, c={completion_tokens}): {response_text}")

        # --- Parse Intent and Response ---
        intent = ""
        direct_response = ""
        try:
            response_data = json.loads(response_text)
            if isinstance(response_data, dict):
                intent = response_data.get("intent", "").lower() if response_data.get("intent") else ""
                direct_response = response_data.get("response", "") if response_data.get("response") else ""
                valid_intents = ["menu", "order", "greeting", "end", "irrelevant", "history", "order_confirmation"]
                if not intent or intent not in valid_intents:
                    logger.warning(f"Invalid or missing intent '{intent}', defaulting to 'irrelevant'")
                    intent = "irrelevant"
                    if not direct_response: direct_response = "I'm sorry, I encountered an issue. I can only assist with menu questions and food orders."
                if intent in ["greeting", "irrelevant", "history"]:
                   if not isinstance(direct_response, str) or not direct_response.strip() or direct_response.strip() in ['{', '}', '[]', '[', ']', '{}', ':', '""', "''", ',', '.']:
                       logger.warning(f"Invalid/empty/fragment direct response for intent '{intent}': '{direct_response}'. Using fallback.")
                       fallback_map = {
                           "greeting": "Hello! How can I help with the menu or your order?",
                           "history": "I can see you're asking about our previous conversation. How can I help you with our menu or placing an order?",
                           "irrelevant": "I'm sorry, I can only assist with questions about our menu and help you place an order."
                       }
                       direct_response = fallback_map.get(intent, "I'm sorry, I can only assist with questions about our menu and help you place an order.")
                logger.info(f"Parsed JSON response: intent='{intent}', response_length={len(direct_response)}")
            else:
                logger.error("Router response data is not a dictionary, defaulting to irrelevant")
                intent = "irrelevant"
                direct_response = "I'm sorry, I encountered an issue processing the response. I can only assist with menu questions and food orders."
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse router JSON response: {e}")
            logger.info(f"Raw response text was: {response_text}")
            logger.info("Attempting keyword-based intent extraction as fallback...")
            raw_text_lower = response_text.lower()
            if "menu" in raw_text_lower: intent = "menu"
            elif any(kw in raw_text_lower for kw in ["order", "cart", "checkout"]): intent = "order"
            elif any(kw in raw_text_lower for kw in ["confirm", "yes", "place my order", "nothing else", "that's all", "i'm done"]): intent = "order_confirmation"
            elif any(kw in raw_text_lower for kw in ["hello", "hi ", " how are"]):
                 intent = "greeting"
                 direct_response = "Hello! How can I help you with the menu or your order today?"
            elif any(kw in raw_text_lower for kw in ["history", "what did i ask", "previous message"]): intent = "history"
            elif any(farewell in raw_text_lower for farewell in ["bye", "thank you", "thanks"]): intent = "end"
            else: intent = "irrelevant"
            if intent != "greeting" and not direct_response:
                direct_response = "I'm sorry, I had trouble understanding that. I can only assist with menu questions and food orders."
            logger.info(f"Fallback intent detection: '{intent}'")

        logger.info(f"Intent classified as: '{intent}' (took {elapsed:.2f}s)")

        # --- Decide Next Step ---
        response = ""
        action_type = ""
        result = None

        try:
            if intent == "menu":
                logger.info("Intent: MENU. Returning acknowledgment.")
                response = "Give us a moment while we research that for you."
                action_type = "menu_inquiry_pending"
                result = ResponseEvent(
                    response=response,
                    action_type=action_type,
                    original_query=query,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
            elif intent == "order":
                logger.info("Intent: ORDER. Returning acknowledgment.")
                if any(kw in query.lower() for kw in ["change", "modify", "add", "remove", "update"]):
                    response = "Give us a moment while we get that order modification ready for you."
                else:
                    response = "Give us a moment while we get that order ready for you."
                action_type = "order_action_pending"
                result = ResponseEvent(
                    response=response,
                    action_type=action_type,
                    original_query=query,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
            elif intent == "order_confirmation":
                logger.info("Intent: ORDER_CONFIRMATION. Returning acknowledgment.")
                response = "Confirming your order..."
                action_type = "order_confirmation_pending"
                result = ResponseEvent(
                    response=response,
                    action_type=action_type,
                    original_query=query,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
            elif intent == "greeting":
                logger.info("Handling greeting directly")
                response = await format_response_text(direct_response)
                action_type = "greeting"
                result = ChatResponseStopEvent(
                    result=None, response=response, action_type=action_type,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                )
            elif intent == "end":
                logger.info("Handling end conversation")
                end_response_text, end_p_tokens, end_c_tokens = await handle_end_conversation(query, self.chat_history)
                response = await format_response_text(end_response_text)
                action_type = "end_conversation"
                result = ChatResponseStopEvent(
                    result=None, response=response, action_type=action_type,
                    prompt_tokens=end_p_tokens, completion_tokens=end_c_tokens
                )
            elif intent == "history":
                logger.info("Handling history query directly")
                response = await format_response_text(direct_response)
                action_type = "history_query"
                result = ChatResponseStopEvent(
                    result=None, response=response, action_type=action_type,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                 )
            elif intent == "irrelevant":
                 logger.info("Handling irrelevant query directly")
                 response = await format_response_text(direct_response)
                 action_type = "irrelevant_query"
                 result = ChatResponseStopEvent(
                    result=None, response=response, action_type=action_type,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                 )
            else:
                logger.error(f"Reached unexpected else block for intent: {intent}")
                response = "I'm sorry, I'm not sure how to handle that. I can assist with menu questions and orders."
                action_type = "error"
                result = ChatResponseStopEvent(
                    result=None, response=response, action_type=action_type,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                )

            logger.info(f"Step 1 Result: Type='{action_type}', Response='{response[:50]}...', P Tokens={getattr(result, 'prompt_tokens', 'N/A')}, C Tokens={getattr(result, 'completion_tokens', 'N/A')}")
            return result

        except Exception as e:
            logger.error(f"Error in classify_and_respond logic block: {type(e).__name__}: {str(e)}")
            simple_error_response = "I'm sorry, I encountered an error and cannot process your request. I can only assist with menu questions and food orders."
            return ChatResponseStopEvent(
                result=None, response=simple_error_response, action_type="error",
                prompt_tokens=None, completion_tokens=None
            )

    @step
    async def generate_detailed_response(self, ctx: Context, ev: ResponseEvent) -> ResponseEvent:
        """
        Second step: If the previous step returned a pending action type,
        generate the detailed response using the appropriate **external handler**.
        Passes the event through if the action type is already final.
        """
        logger.info(f"Entering generate_detailed_response with action_type: {ev.action_type}")

        response_text = ev.response
        final_action_type = ev.action_type
        cart_items = ev.cart_items
        cart_status = ev.cart_status
        prompt_tokens = ev.prompt_tokens
        completion_tokens = ev.completion_tokens

        if ev.action_type == "menu_inquiry_pending":
            logger.info("Handling pending menu inquiry via external handler")
            if ev.original_query:
                response_text, p_tokens, c_tokens = await handle_menu_query(ev.original_query, self.chat_history, self.menu_text)
                formatted_response = await format_response_text(response_text)
                prompt_tokens, completion_tokens = p_tokens, c_tokens
                final_action_type = "menu_inquiry"
                logger.info(f"Generated detailed menu response: {formatted_response[:50]}...")
                response_text = formatted_response
            else:
                logger.error("Original query missing for menu_inquiry_pending")
                response_text = "Error: Missing query for menu info."
                final_action_type = "error"

        elif ev.action_type == "order_action_pending":
            logger.info("Handling pending order action via external handler")
            if ev.original_query:
                response_text, cart, p_tokens, c_tokens = await handle_order_query(ev.original_query, self.chat_history, self.menu_text)
                formatted_response = await format_response_text(response_text)
                cart_items = cart
                prompt_tokens, completion_tokens = p_tokens, c_tokens
                final_action_type = "order_action"
                logger.info(f"Generated detailed order response: {formatted_response[:50]}...")
                response_text = formatted_response
            else:
                logger.error("Original query missing for order_action_pending")
                response_text = "Error: Missing query for order action."
                final_action_type = "error"

        elif ev.action_type == "order_confirmation_pending":
            logger.info("Handling pending order confirmation via external handler")
            if ev.original_query:
                response_text, cart, status, p_tokens, c_tokens = await handle_order_confirmation(ev.original_query, self.chat_history)
                formatted_response = await format_response_text(response_text)
                cart_items = cart
                cart_status = status
                prompt_tokens, completion_tokens = p_tokens, c_tokens
                final_action_type = "order_confirmation"
                logger.info(f"Generated order confirmation response: {formatted_response[:50]}...")
                response_text = formatted_response
            else:
                logger.error("Original query missing for order_confirmation_pending")
                response_text = "Error: Missing query for order confirmation."
                final_action_type = "error"

        elif ev.action_type in ["greeting", "end_conversation", "history_query", "irrelevant_query", "error"]:
            logger.info(f"Passing through response with final action_type: {ev.action_type}")
            response_text = ev.response
            final_action_type = ev.action_type
            cart_items = ev.cart_items
            cart_status = ev.cart_status
            prompt_tokens = ev.prompt_tokens
            completion_tokens = ev.completion_tokens
        else:
            logger.warning(f"generate_detailed_response received unexpected action type: {ev.action_type}. Passing through.")
            response_text = ev.response
            final_action_type = ev.action_type
            cart_items = ev.cart_items
            cart_status = ev.cart_status
            prompt_tokens = ev.prompt_tokens
            completion_tokens = ev.completion_tokens

        return ResponseEvent(
            response=response_text,
            action_type=final_action_type,
            original_query=ev.original_query,
            cart_items=cart_items,
            cart_status=cart_status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )

    @step
    async def finalize(self, ctx: Context, ev: ResponseEvent) -> ChatResponseStopEvent:
        """
        Final step: Convert the final ResponseEvent to ChatResponseStopEvent.
        The ResponseEvent now contains the final state after detailed processing (if needed).
        """
        logger.info(f"Finalizing response: {ev.response[:30]}...")

        prompt_tokens = ev.prompt_tokens
        completion_tokens = ev.completion_tokens

        logger.info(f"Finalize using token values: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")

        result = ChatResponseStopEvent(
            result=None,
            response=ev.response,
            action_type=ev.action_type,
            cart_items=ev.cart_items,
            cart_status=ev.cart_status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        logger.info(f"Created ChatResponseStopEvent with fields: response={result.response[:20]}..., action_type={result.action_type}, p_tokens={result.prompt_tokens}, c_tokens={result.completion_tokens}")
        return result 