import logging
import time
import json
from typing import List, Tuple, Optional, Dict, Any
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.llms.openai import OpenAI

# Set up logger (or pass it in)
logger = logging.getLogger("food_ordering_bot.handlers")

# --- Token Extraction Helper ---

def _extract_token_counts(response: Any, handler_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Extracts prompt and completion tokens from an OpenAI response object."""
    prompt_tokens = None
    completion_tokens = None
    try:
        if hasattr(response, 'additional_kwargs') and isinstance(response.additional_kwargs, dict):
            kwargs_dict = response.additional_kwargs
            if 'prompt_tokens' in kwargs_dict:
                p_tokens = kwargs_dict['prompt_tokens']
                if isinstance(p_tokens, str) and p_tokens.isdigit():
                    prompt_tokens = int(p_tokens)
                elif isinstance(p_tokens, int):
                    prompt_tokens = p_tokens
            if 'completion_tokens' in kwargs_dict:
                c_tokens = kwargs_dict['completion_tokens']
                if isinstance(c_tokens, str) and c_tokens.isdigit():
                    completion_tokens = int(c_tokens)
                elif isinstance(c_tokens, int):
                    completion_tokens = c_tokens
        else:
            logger.debug(f"Token Extraction ({handler_name}): No valid additional_kwargs found.")
    except Exception as token_err:
        logger.error(f"Error extracting token counts for {handler_name}: {token_err}")
        prompt_tokens = None
        completion_tokens = None
    return prompt_tokens, completion_tokens

# --- Formatting Handler ---

async def format_response_text(text: str) -> str:
    """
    Uses an LLM to format the text response for consistent, plain text output,
    ensuring lists are lettered (A, B, C...) and suitable for SMS/basic chat.
    Removes extraneous markdown. Returns original text on error.
    """
    logger.info(f"Formatting text (input): {text[:100]}...")

    if not text or text.isspace():
        logger.info("Formatting text (input was empty/whitespace): Returning empty string.")
        return ""

    format_prompt = f"""
    You are a text formatting assistant. Your task is to take the input text, clean it up, and format it for display in a simple text-based chat interface (like SMS).

    **Formatting Rules (Apply these STRICTLY):**
    1.  **Plain Text:** The final output MUST be plain text only. Remove ALL markdown formatting (like **, *, _, etc.). Exception: Keep existing newline characters (`\n`) where they make sense for readability.
    2.  **List Formatting:** Identify any lists of choices or options presented to the user. Reformat these lists so EACH item starts with a capital letter followed by a period and a space (A., B., C., etc.). Each item MUST be on its own line. Make sure the original item text follows the letter label. Preserve any existing list structure if it already uses letter labels correctly, just clean markdown.
    3.  **Clarity:** Ensure the text is clear and easy to read. Do not add any conversational text, greetings, questions, or suggestions that were not present in the original input. Your ONLY job is to clean and reformat the *existing* text according to the rules.
    4.  **No Extra Content:** Do not add headers, footers, or any text not derived from the original input.
    5.  **Whitespace:** Trim leading/trailing whitespace from the final output. Preserve internal newlines essential for structure (like between list items or paragraphs). Ensure consistent line breaks around list items.

    **Example Input 1 (with markdown and bad list):**
    "Okay, we have the *Classic Chicken Sandwich* ($8.99) and the **Spicy Deluxe** ($9.99). Would you like one?\n- Option 1: Classic\n- Option 2: Spicy"

    **Example Output 1 (formatted):**
    "Okay, we have the Classic Chicken Sandwich ($8.99) and the Spicy Deluxe ($9.99). Would you like one?
    A. Classic
    B. Spicy"

    **Example Input 2 (already somewhat formatted but needs standardization):**
    "We have these pizza options:\nA. Pepperoni Pizza ($12.00)\nB. Margherita Pizza ($11.00)\nWould you like to add one?"

    **Example Output 2 (standardized):**
    "We have these pizza options:
    A. Pepperoni Pizza ($12.00)
    B. Margherita Pizza ($11.00)
    Would you like to add one?"

    **Example Input 3 (Confirmation prompt):**
    "Your order:\n- 1 Classic Chicken Sandwich (extra cheese): $9.99\n\nTotal: $9.99\n\nWould you like to confirm this order?\nA. Yes, confirm my order\nB. No, I'd like to make changes"

    **Example Output 3 (already correct, should pass through cleaned):**
    "Your order:
    - 1 Classic Chicken Sandwich (extra cheese): $9.99

    Total: $9.99

    Would you like to confirm this order?
    A. Yes, confirm my order
    B. No, I'd like to make changes"

    **Input Text to Format:**
    ---
    {text}
    ---

    **Formatted Output:**
    """

    llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=20)
    try:
        response = await llm.acomplete(format_prompt)
        formatted_text = response.text.strip()
        logger.info(f"Formatting text (output): {formatted_text[:100]}...")

        if not formatted_text and text and not text.isspace():
            logger.warning("Formatter LLM returned empty string unexpectedly. Falling back to original text.")
            return text

        return formatted_text
    except Exception as e:
        logger.error(f"Error during text formatting LLM call: {type(e).__name__}: {str(e)}. Returning original text.")
        return text

# --- Intent-Specific Handlers ---

async def handle_menu_query(
    query: str,
    chat_history: List[ChatMessage],
    menu_text: str
) -> Tuple[str, Optional[int], Optional[int]]:
    """Handle menu-related queries. Returns (response_text, prompt_tokens, completion_tokens)."""
    menu_template = f"""
    You are a helpful restaurant assistant providing information about menu items and guiding users towards placing an order.
    **Use the provided conversation history to understand the context and avoid repeating information unnecessarily.**

    **Presenting Options:**
    - ALWAYS present menu items and options with letter labels (A, B, C, etc.) at the beginning of each choice.
    - When listing multiple items, present them in a clear lettered list format, with one option per line.
    - Example:
      We have these sandwich options:
      A. Classic Chicken Sandwich ($8.99)
      B. Spicy Deluxe Chicken Sandwich ($9.99)
      C. Grilled Chicken Club ($10.49)

    **Handling Follow-up for "More Details":**
    - If the user asks for "more details" after you've provided a summary, look at your *immediately preceding message* in the history.
    - Identify the items you summarized in that message.
    - Provide the *additional* details (like descriptions, options, ingredients) for *only those items*.
    - Do NOT repeat the item names and prices from the summary unless essential for context.

    **Guiding towards Purchase:**
    - After providing information about an item or category, gently ask if the user would like to add anything to their order or if they need more information.

    Be friendly and informative about prices and descriptions.
    Use standard text formatting. Use letter labels (A, B, C...) for options.
    Do NOT use any markdown (like ** for bolding).
    The complete menu is as follows:
    {menu_text}

    **Example Interaction (Presenting Options):**
    User: "What kind of pizzas do you have?"
    Assistant: "We have these pizza options:
    A. Pepperoni Pizza ($12.00)
    B. Margherita Pizza ($11.00)
    C. Veggie Pizza ($11.50)
    Would you like to add one to your order or hear more details about any of these options?"
    """

    messages = chat_history + [
        ChatMessage(role=MessageRole.USER, content=query),
        ChatMessage(role=MessageRole.SYSTEM, content=menu_template)
    ]

    logger.info("handle_menu_query: Sending request to OpenAI")
    llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=30)
    prompt_tokens, completion_tokens = None, None
    try:
        start_time = time.time()
        response = await llm.achat(messages)
        elapsed = time.time() - start_time
        prompt_tokens, completion_tokens = _extract_token_counts(response, "handle_menu_query")
        logger.info(f"handle_menu_query: Got response in {elapsed:.2f}s (prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens})")
        return response.message.content, prompt_tokens, completion_tokens
    except Exception as e:
        logger.error(f"handle_menu_query: Error: {type(e).__name__}: {str(e)}")
        return f"I'm sorry, I had trouble providing menu information. Error: {str(e)}", prompt_tokens, completion_tokens

async def handle_order_query(
    query: str,
    chat_history: List[ChatMessage],
    menu_text: str
) -> Tuple[str, List[Dict[str, Any]], Optional[int], Optional[int]]:
    """Handle order-related actions. Returns (response_text, cart_items, prompt_tokens, completion_tokens)."""
    order_template = f"""
    You are an assistant helping with food orders and guiding the user towards completing their purchase.
    **Use the provided conversation history to understand the current order status and context.**
    Be clear about prices and options. Summarize complex orders or options if necessary.
    Offer to provide more detail if needed.
    If they want something not on the menu, politely inform them it's unavailable.

    **Presenting Options (General):**
    - ALWAYS present choices with letter labels (A, B, C, etc.) at the beginning of each choice.
    - This applies to menu items, customization options, and any selection the user needs to make.
    - List each option on a new line.
    - Use standard text formatting only. Do NOT use markdown (like ** for bolding).
    - Example format for presenting drink sizes:
      Would you like:
      A. Small Coke ($2.00)
      B. Medium Coke ($2.50)
      C. Large Coke ($3.25)

    **Presenting Options (Item-Specific Add-ons/Modifications):**
    - **If the user adds an item that has specific options listed in the menu (like add cheese, add bacon, make it spicy for a sandwich):**
        - First, confirm the item was added (e.g., "Great choice! Added Classic Chicken Sandwich to your cart.").
        - Then, ask if they want to add any of *its specific options*.
        - List these specific options clearly WITH LETTER LABELS (A, B, C, etc.), including the price modifier (e.g., `+$.50`, `(no charge)`). Use plain text.
        - Example:
          "Would you like to add any options?
          A. Add Cheese (+$1.00)
          B. Add Bacon (+$1.50)
          C. Make It Spicy (+$0.50)
          D. Substitute Grilled Chicken (no charge)
          Let me know if you want to add options, add more items, or checkout!"
    - **If the user adds an item with NO specific options listed in the menu:**
         - Simply confirm the item was added (e.g., "Okay, Regular Fries added.") and ask if they want to add anything else or checkout. Use plain text.

    In addition to your text response, you must also manage and return the user's cart state.
    You need to parse the user's intent and:
    1. For "add" - add items to the cart
    2. For "remove" - remove items from the cart
    3. For "change" - modify existing items (e.g., change quantity, options)
    4. For "upgrade" - upgrade items (e.g., size, add-ons)
    5. For "cancel order" - empty the cart
    6. For "make order" / "checkout" / "confirm" - This should be handled by the ORDER_CONFIRMATION intent, but acknowledge if the user explicitly mentions it here and potentially ask if they are ready to confirm.

    When responding, output BOTH:
    1. A conversational text message acknowledging the user's action.
       - After modifying the cart (add/remove/change), confirm the current state of the cart and ask if they want to add anything else or proceed to checkout. Use plain text.
    2. A valid JSON representation of their updated cart

    The cart should be a JSON array of objects with properties:
    - "item": string - the menu item name
    - "quantity": number - how many of this item
    - "options": array of strings - any options/modifications
    - "price": number - the unit price of this item including options

    FORMAT:
    {{
      "response": "Your plain text natural language response here. Guide towards checkout if appropriate.",
      "cart": [updated cart items]
    }}

    Based on the menu information and the user's request, help them place or modify their order.
    The complete menu is as follows:
    {menu_text}

    **Example Interaction (Item with Options):**
    User: "Can I have a Classic Chicken Sandwich?"
    Assistant Output (JSON):
    {{
      "response": "Great choice! Added Classic Chicken Sandwich ($8.99) to your cart. Would you like to add any options?\nA. Add Cheese (+$1.00)\nB. Add Bacon (+$1.50)\nC. Make It Spicy (+$0.50)\nD. Substitute Grilled Chicken (no charge)\nOr let me know if you want to add more items or checkout!",
      "cart": [{{"item": "Classic Chicken Sandwich", "quantity": 1, "price": 8.99, "options": []}}]
    }}

    **Example Interaction (General Choice - Size):**
    User: "Add a coke"
    Assistant Output (JSON):
    {{
      "response": "Sure thing. We have:\nA. Small Coke ($2.00)\nB. Medium Coke ($2.50)\nC. Large Coke ($3.25)\nWhich one would you like?",
      "cart": [/* existing cart items */]
    }}
    """

    messages = chat_history + [
        ChatMessage(role=MessageRole.USER, content=query),
        ChatMessage(role=MessageRole.SYSTEM, content=order_template)
    ]

    logger.info("handle_order_query: Sending request to OpenAI")
    llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=30)
    prompt_tokens, completion_tokens = None, None
    cart_items = []
    response_content = ""
    try:
        start_time = time.time()
        response = await llm.achat(messages)
        elapsed = time.time() - start_time
        prompt_tokens, completion_tokens = _extract_token_counts(response, "handle_order_query")
        logger.info(f"handle_order_query: Got response in {elapsed:.2f}s (prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens})")

        response_content = response.message.content

        try:
            if isinstance(response_content, str):
                import re
                json_matches = re.findall(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', response_content, re.DOTALL)
                if json_matches:
                    for json_str in json_matches:
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict) and "response" in data:
                                response_content = data.get("response", "")
                                if "cart" in data and isinstance(data["cart"], list):
                                    cart_items = data["cart"]
                                    logger.info(f"Extracted cart items: {len(cart_items)} items")
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    logger.warning("No JSON objects found in the order response")
            else:
                logger.warning(f"Order response content is not a string: {type(response_content)}")
        except Exception as e:
            logger.error(f"Error extracting cart data from order response: {type(e).__name__}: {str(e)}")

        return response_content, cart_items, prompt_tokens, completion_tokens
    except Exception as e:
        logger.error(f"handle_order_query: Error: {type(e).__name__}: {str(e)}")
        return f"I'm sorry, I had trouble with your order. Error: {str(e)}", cart_items, prompt_tokens, completion_tokens

async def handle_end_conversation(
    query: str,
    chat_history: List[ChatMessage]
) -> Tuple[str, Optional[int], Optional[int]]:
    """Handle end of conversation. Returns (response_text, prompt_tokens, completion_tokens)."""
    end_template = """
    The user seems to be ending the conversation.
    **Consider the conversation history for context if appropriate (e.g., thanking them for an order).**
    Respond with a friendly, concise, **plain text** goodbye message that invites them to return. Do not use markdown.
    """

    messages = chat_history + [
        ChatMessage(role=MessageRole.USER, content=query),
        ChatMessage(role=MessageRole.SYSTEM, content=end_template)
    ]

    logger.info("handle_end_conversation: Sending request to OpenAI")
    llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=30)
    prompt_tokens, completion_tokens = None, None
    try:
        start_time = time.time()
        response = await llm.achat(messages)
        elapsed = time.time() - start_time
        prompt_tokens, completion_tokens = _extract_token_counts(response, "handle_end_conversation")
        logger.info(f"handle_end_conversation: Got response in {elapsed:.2f}s (prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens})")
        return response.message.content, prompt_tokens, completion_tokens
    except Exception as e:
        logger.error(f"handle_end_conversation: Error: {type(e).__name__}: {str(e)}")
        return f"I'm sorry, I had trouble saying goodbye. Error: {str(e)}", prompt_tokens, completion_tokens

async def handle_order_confirmation(
    query: str,
    chat_history: List[ChatMessage]
) -> Tuple[str, List[Dict[str, Any]], Optional[str], Optional[int], Optional[int]]:
    """Handle order confirmation. Returns (response_text, cart_items, cart_status, prompt_tokens, completion_tokens)."""
    extracted_cart_items = []
    try:
        for msg in reversed(chat_history):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                import re
                json_matches = re.findall(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', msg.content, re.DOTALL)
                for json_str in json_matches:
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, dict) and "cart" in data and isinstance(data["cart"], list):
                            extracted_cart_items = data["cart"]
                            break
                    except json.JSONDecodeError:
                        continue
            if extracted_cart_items: break
    except Exception as e:
        logger.error(f"Error extracting cart from history for confirmation: {str(e)}")
        extracted_cart_items = []

    validated_cart_items = []
    for item in extracted_cart_items:
        if isinstance(item, dict):
            if "item" not in item: item["item"] = "Unknown item"
            if "quantity" not in item: item["quantity"] = 1
            if "price" not in item: item["price"] = 0.0
            if "options" not in item or not isinstance(item["options"], list): item["options"] = []
            validated_cart_items.append(item)
        else:
            logger.warning(f"Skipping invalid cart item during confirmation init: {item}")
    initial_cart_items = validated_cart_items

    confirmation_template = f"""
    You are a helpful restaurant assistant confirming an order.

    **User's current message:** "{query}"

    **Order confirmation context:**
    When the user says messages like "nothing else", "that's all", "I'm done", or "checkout", this means they want to finish ordering and are ready to confirm their order. If this is the case, summarize their order and ask for final confirmation.

    When the user says things like "yes", "confirm", "approved", "correct", or similar phrases, they are confirming their order. If you determine this is a confirmation, mark the order as CONFIRMED and thank them.

    **Instructions:**
    1. If the cart is empty, tell them they need to add items first.
    2. If there are items in the cart and the user is finishing their order without explicitly confirming it:
       - Start the response with "Your order:".
       - List EACH item from the `cart`. Use plain text (no markdown). Include quantity, any options (concisely), and the item's calculated price (quantity * unit price). Use a bulleted list (`- ` or `* `) for the items.
       - After listing ALL items, include the total price (e.g., "Total: $XX.XX").
       - Ask for confirmation with LETTER CHOICES: "Would you like to confirm this order?\nA. Yes, confirm my order\nB. No, I'd like to make changes"
       - Set status to "PENDING CONFIRMATION".
    3. If the user is explicitly confirming a previous confirmation request, thank them, set status to "CONFIRMED". Use plain text.

    FORMAT (empty cart):
    {{
      "response": "Your cart is empty. Please add items before confirming your order.",
      "cart": [],
      "cart_status": "OPEN"
    }}

    FORMAT (items in cart, user is finishing order but hasn't confirmed):
    {{
      "response": "Your order:\n- 1 Classic Chicken Sandwich (extra cheese): $9.99\n- 2 Sodas (Large): $6.50\n\nTotal: $16.49\n\nWould you like to confirm this order?\nA. Yes, confirm my order\nB. No, I'd like to make changes",
      "cart": [existing cart items],
      "cart_status": "PENDING CONFIRMATION"
    }}

    FORMAT (user confirming order):
    {{
      "response": "Thank you! Your order has been confirmed and will be ready shortly.",
      "cart": [existing cart items],
      "cart_status": "CONFIRMED"
    }}

    Based on the cart contents and user message, provide the appropriate plain text response. Ensure the total price is calculated correctly.
    IMPORTANT: Use the initial cart provided as the basis for response generation. Return the correct cart and status in the final JSON.
    Initial Cart State for Context: {json.dumps(initial_cart_items)}
    """

    messages = chat_history + [
        ChatMessage(role=MessageRole.USER, content=query),
        ChatMessage(role=MessageRole.SYSTEM, content=confirmation_template)
    ]

    logger.info("handle_order_confirmation: Sending request to OpenAI")
    llm = OpenAI(model="gpt-4o", temperature=0.0, request_timeout=30)
    prompt_tokens, completion_tokens = None, None
    response_content = ""
    final_cart_items = initial_cart_items.copy()
    cart_status = "OPEN"

    try:
        start_time = time.time()
        response = await llm.achat(messages)
        elapsed = time.time() - start_time
        prompt_tokens, completion_tokens = _extract_token_counts(response, "handle_order_confirmation")
        logger.info(f"handle_order_confirmation: Got response in {elapsed:.2f}s (prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens})")

        response_content = response.message.content

        try:
            if isinstance(response_content, str):
                import re
                json_matches = re.findall(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', response_content, re.DOTALL)
                if json_matches:
                    for json_str in json_matches:
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict) and "response" in data:
                                response_content = data.get("response", "")
                                if "cart" in data and isinstance(data["cart"], list):
                                    extracted_cart = data["cart"]
                                    validated_extracted_cart = []
                                    for item in extracted_cart:
                                        if isinstance(item, dict):
                                            if "item" not in item: item["item"] = "Unknown item"
                                            if "quantity" not in item: item["quantity"] = 1
                                            if "price" not in item: item["price"] = 0.0
                                            if "options" not in item or not isinstance(item["options"], list): item["options"] = []
                                            validated_extracted_cart.append(item)
                                        else:
                                            logger.warning(f"Skipping invalid extracted cart item in confirmation: {item}")
                                    final_cart_items = validated_extracted_cart
                                    logger.info(f"Extracted cart items in confirmation: {len(final_cart_items)} items")
                                if "cart_status" in data and isinstance(data["cart_status"], str):
                                    cart_status = data["cart_status"]
                                    logger.info(f"Extracted cart status: {cart_status}")
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    logger.warning("No JSON objects found in the confirmation response")
            else:
                logger.warning(f"Confirmation response content is not a string: {type(response_content)}")
        except Exception as e:
            logger.error(f"Error extracting cart/status data from confirmation response: {type(e).__name__}: {str(e)}")

        return response_content, final_cart_items, cart_status, prompt_tokens, completion_tokens
    except Exception as e:
        logger.error(f"handle_order_confirmation: Error: {type(e).__name__}: {str(e)}")
        return f"I'm sorry, I had trouble confirming your order. Error: {str(e)}", final_cart_items, cart_status, prompt_tokens, completion_tokens 