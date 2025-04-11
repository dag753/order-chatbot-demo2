import streamlit as st
import os
import asyncio
import logging
import sys
import time
import json
from dotenv import load_dotenv
from llama_index.core.workflow import StartEvent, StopEvent
from chat_engine import create_chat_engine, ChatResponseStopEvent
from utils import initialize_session_state
from ui_components import render_sidebar, display_chat_messages # Remove display_cart import
import re # Import regex module

# Set up logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("food_ordering_bot")
logger.setLevel(logging.INFO) # Use INFO for less noise, DEBUG is very verbose

# Load environment variables
load_dotenv()

# --- Helper function for cleaning response text ---
# This function is no longer used and will be removed.

async def process_message(workflow, user_query):
    """Process a message through the workflow and return the response"""
    try:
        # Measure total workflow time (primarily for stage 1 or simple intents)
        total_start_time = time.time()
        
        # Create a start event
        start_event = StartEvent(content=user_query)
        logger.info("Debug: Created StartEvent")
        
        # Attempt to run workflow
        logger.info("Debug: About to run workflow")
        try:
            result = await workflow.run(start_event=start_event)
        except Exception as workflow_error:
            logger.error(f"Workflow execution error: {type(workflow_error).__name__}: {str(workflow_error)}", exc_info=True)
            # Create a fallback response for workflow errors
            return ChatResponseStopEvent(
                result=None,
                response=f"I'm sorry, I encountered an error while processing your request. Please try again with different wording.",
                action_type="error",
                cart_items=None
            )
        
        # Calculate and log total workflow time
        total_elapsed = time.time() - total_start_time
        logger.info(f"==== Workflow processing time (Stage 1 / Full): {total_elapsed:.2f}s ====")
        
        # Log the result type and str representation
        result_type = type(result).__name__ if result is not None else "None"
        logger.info(f"Debug: Workflow result type: {result_type}")
        
        # The workflow should return a ChatResponseStopEvent containing our ResponseEvent
        if isinstance(result, ChatResponseStopEvent):
            logger.info(f"Debug: Got ChatResponseStopEvent with response: {result.response[:30]}...")
            logger.info(f"Debug: Action type: {result.action_type}")
            return result # Return the whole ChatResponseStopEvent
        elif result is None:
            logger.error("Workflow returned None unexpectedly")
            # Create a fallback StopEvent
            return ChatResponseStopEvent(
                result=None,
                response="I'm sorry, but something went wrong with my processing. The workflow returned no response.",
                action_type="error",
                cart_items=None
            )
        elif isinstance(result, StopEvent):
            # Handle case where we get a regular StopEvent (shouldn't happen with our setup)
            logger.warning(f"Debug: Got regular StopEvent instead of ChatResponseStopEvent: {result}")
            return ChatResponseStopEvent(
                result=None,
                response=str(result.result) if result.result else "No response available from StopEvent",
                action_type="unknown",
                cart_items=None
            )
        else:
            # Unexpected result type
            logger.error(f"Debug: Unexpected result type from workflow: {result_type}")
            return ChatResponseStopEvent(
                result=None,
                response=f"Got unexpected result type: {result_type}",
                action_type="error",
                cart_items=None
            )
        
    except Exception as e:
        # Log the full exception
        logger.error(f"Error in process_message: {type(e).__name__}: {str(e)}", exc_info=True)
        # Make sure we return something valid here
        return ChatResponseStopEvent(
            result=None,
            response=f"I'm sorry, I couldn't process your request. Please try again with different wording.",
            action_type="error",
            cart_items=None
        )

async def handle_chat_submission(prompt: str):
    """Handles the logic after a user submits a message."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=False)

    # Log user action
    st.session_state.actions.append(f"User said: {prompt}")

    # --- Stage 1: Get Initial Response ---
    chat_message_placeholder = st.chat_message("assistant")

    # Create workflow instance
    chat_workflow = create_chat_engine(
        st.session_state.menu,
        st.session_state.messages[:-1] # Exclude current user prompt
    )

    # Process message (Stage 1)
    logger.info(f"Processing user message: {prompt[:30]}...")
    stage1_start_time = time.time()
    response_event = await process_message(chat_workflow, prompt) # Changed to await
    stage1_time = time.time() - stage1_start_time
    logger.info(f"====== WORKFLOW PROCESSING TIME (Stage 1 / Full): {stage1_time:.2f}s ======")
    logger.info(f"Stage 1 ResponseEvent received: {response_event}")

    # --- Display Initial Response / Handle Immediate Errors ---
    initial_message_content = ""
    action_type = "error" # Default if response_event is None or lacks type
    initial_response_handled = False # Flag
    cart_items = None # Initialize cart_items as None
    cart_status = None # Initialize cart_status as None

    try:
        if response_event is None or not isinstance(response_event, ChatResponseStopEvent):
            initial_message_content = "Sorry, I couldn't process your request right now (Invalid workflow response). Please try again later."
            logger.error(f"Invalid or None response received from process_message: {response_event}")
        else:
            action_type = response_event.action_type
            initial_message_content = response_event.response
            # Extract cart items if available
            cart_items = response_event.cart_items
            # Extract cart status if available
            cart_status = response_event.cart_status
            
            logger.info(f"Initial response text (first 50 chars): {initial_message_content[:50]}...")
            
            # --- Basic validation and CLEANING ---
            if not isinstance(initial_message_content, str) or not initial_message_content.strip():
                initial_message_content = "I'm sorry, I didn't generate a proper initial response. Please try again."
                logger.warning(f"Found empty response: {response_event.response}")
                action_type = "error" # Treat as error
            else:
                # Check for problematic fragments after cleaning (or lack thereof)
                if not initial_message_content or initial_message_content.strip() in ['{', '}', '[]', '[', ']', '{}']:
                    initial_message_content = "I'm sorry, I didn't generate a proper initial response. Please try again."
                    logger.warning(f"Found invalid fragment before/without cleaning: {response_event.response}")
                    action_type = "error"
                # Apply basic text cleaning in case formatting wasn't done upstream
                elif isinstance(initial_message_content, str):
                    pass # Cleaning is likely interfering with intended LLM markdown formatting
        
        # Display initial message (or error) in the first placeholder
        with chat_message_placeholder:
             st.markdown(initial_message_content, unsafe_allow_html=False)
             # Only show time if it's positive
             if stage1_time > 0:
                 # Include token counts if available
                 prompt_tokens = getattr(response_event, 'prompt_tokens', None)
                 completion_tokens = getattr(response_event, 'completion_tokens', None)
                 token_info = ""
                 if prompt_tokens is not None and completion_tokens is not None:
                     token_info = f", prompt: {prompt_tokens}, completion: {completion_tokens}"
                 st.caption(f"*Response time: {stage1_time:.2f}s{token_info}*")

        # Add this first message to history AFTER displaying it
        st.session_state.messages.append({"role": "assistant", "content": initial_message_content})
        # Add action to log
        st.session_state.actions.append(f"Action: {action_type}")
        # Store response time mapping (using index of the message just added)
        message_index = len(st.session_state.messages) - 1
        st.session_state.response_times[message_index] = {
            "time": stage1_time,
            "prompt_tokens": getattr(response_event, 'prompt_tokens', None),
            "completion_tokens": getattr(response_event, 'completion_tokens', None)
        }
        initial_response_handled = True # Mark that *something* was shown
        
        # Update cart state if cart items were returned (even if empty - empty cart is valid)
        if cart_items is not None:
            st.session_state.current_cart = cart_items
            if action_type == "order_action" or action_type == "order_confirmation":
                if not cart_items:
                    st.session_state.actions.append("Cart: Order canceled/emptied")
                else:
                    st.session_state.actions.append(f"Cart: Updated with {len(cart_items)} items")
                    
        # Update cart status if provided
        if cart_status is not None:
            st.session_state.cart_status = cart_status
            st.session_state.actions.append(f"Cart Status: Changed to {cart_status}")

    except Exception as e:
        # Catch errors during initial display/handling
        logger.error(f"Error processing/displaying initial response: {e}", exc_info=True)
        if not initial_response_handled: # Only display error if nothing else was shown
            error_message = "Sorry, there was a problem displaying the initial response."
            with chat_message_placeholder:
                st.markdown(error_message, unsafe_allow_html=False)
            # Add error to history if not already handled
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != error_message:
                 st.session_state.messages.append({"role": "assistant", "content": error_message})
                 st.session_state.actions.append("Action: display_error")

    # --- Stage 2: Handle Pending Actions (if applicable) ---
    if action_type in ["menu_inquiry_pending", "order_action_pending", "order_confirmation_pending"]:
        logger.info(f"Handling pending action: {action_type}")
        # Create placeholder for the second message
        stage2_placeholder = st.chat_message("assistant")
        with stage2_placeholder:
            # The spinner now appears *inside* the second message bubble
            with st.spinner("Getting details..."):
                stage2_start_time = time.time()
                detailed_response_content = ""
                final_action_type = "error" # Default for stage 2
                stage2_cart_items = None # For stage 2 cart items
                stage2_cart_status = None # For stage 2 cart status

                try:
                    # Call the appropriate handler directly
                    original_prompt = prompt # Use the user's original prompt
                    stage2_prompt_tokens = None
                    stage2_completion_tokens = None
                    
                    if action_type == "menu_inquiry_pending":
                        detailed_response_content = await chat_workflow._handle_menu_query(original_prompt) # Changed to await
                        # Retrieve tokens from workflow instance AFTER handler call
                        stage2_prompt_tokens = chat_workflow.last_prompt_tokens
                        stage2_completion_tokens = chat_workflow.last_completion_tokens
                        final_action_type = "menu_inquiry"
                    elif action_type == "order_action_pending":
                        detailed_response_content, stage2_cart_items = await chat_workflow._handle_order_query(original_prompt) # Changed to await
                        # Retrieve tokens from workflow instance AFTER handler call
                        stage2_prompt_tokens = chat_workflow.last_prompt_tokens
                        stage2_completion_tokens = chat_workflow.last_completion_tokens
                        final_action_type = "order_action"
                        # Update cart state if cart items were returned
                        if stage2_cart_items is not None:
                            st.session_state.current_cart = stage2_cart_items
                            if not stage2_cart_items:
                                st.session_state.actions.append("Cart: Order canceled/emptied")
                            else:
                                st.session_state.actions.append(f"Cart: Updated with {len(stage2_cart_items)} items")
                    elif action_type == "order_confirmation_pending":
                        detailed_response_content, stage2_cart_items, stage2_cart_status = await chat_workflow._handle_order_confirmation(original_prompt)
                        # Retrieve tokens from workflow instance AFTER handler call
                        stage2_prompt_tokens = chat_workflow.last_prompt_tokens
                        stage2_completion_tokens = chat_workflow.last_completion_tokens
                        final_action_type = "order_confirmation"
                        # Update cart state if cart items were returned
                        if stage2_cart_items is not None:
                            st.session_state.current_cart = stage2_cart_items
                            if not stage2_cart_items:
                                st.session_state.actions.append("Cart: Order canceled/emptied")
                            else:
                                st.session_state.actions.append(f"Cart: Updated with {len(stage2_cart_items)} items")
                                
                        # Update cart status if provided
                        if stage2_cart_status is not None:
                            st.session_state.cart_status = stage2_cart_status
                            st.session_state.actions.append(f"Cart Status: Changed to {stage2_cart_status}")

                    # Basic validation/cleaning
                    if isinstance(detailed_response_content, str) and \
                       (detailed_response_content.strip().startswith('{') and detailed_response_content.strip().endswith('}')):
                        try:
                            json_data = json.loads(detailed_response_content)
                            if isinstance(json_data, dict) and 'response' in json_data:
                                detailed_response_content = json_data['response']
                        except json.JSONDecodeError:
                            pass # Use as is

                    if not detailed_response_content or detailed_response_content.strip() in ['{', '}', '[]', '[', ']', '{}']:
                         detailed_response_content = "I'm sorry, I didn't receive valid details."
                         final_action_type = "error"
                         logger.warning("Received empty/invalid details in stage 2.")
                    
                    # Format the response text using the LLM formatter
                    try:
                        logger.info("Formatting Stage 2 response text with LLM...")
                        detailed_response_content = await chat_workflow._format_response_text(detailed_response_content) # Removed formatting call
                    except Exception as format_err:
                        logger.error(f"Error formatting Stage 2 response: {format_err}")
                        # Continue with unformatted text
                        # Apply basic text cleaning as fallback
                        pass # Formatting failure fallback

                except Exception as e:
                    logger.error(f"Error in stage 2 handling ({action_type}): {e}", exc_info=True)
                    detailed_response_content = "Sorry, I encountered an error while getting the details."
                    final_action_type = "error" # Ensure it's marked as error
                    # Ensure tokens are None if stage 2 failed
                    stage2_prompt_tokens = None
                    stage2_completion_tokens = None 

                stage2_time = time.time() - stage2_start_time
                logger.info(f"Stage 2 processing time: {stage2_time:.2f}s (Tokens: p={stage2_prompt_tokens}, c={stage2_completion_tokens})") # Log retrieved tokens

            # Display detailed response (or error) in the second placeholder AFTER spinner finishes
            st.markdown(detailed_response_content, unsafe_allow_html=False)
            # Only show time if positive
            if stage2_time > 0:
                # Display tokens retrieved from the workflow instance
                token_info = ""
                if stage2_prompt_tokens is not None and stage2_completion_tokens is not None:
                    token_info = f", prompt: {stage2_prompt_tokens}, completion: {stage2_completion_tokens}"
                elif stage2_prompt_tokens is not None: # Handle case where only prompt is available
                     token_info = f", prompt: {stage2_prompt_tokens}"
                elif stage2_completion_tokens is not None: # Handle case where only completion is available
                     token_info = f", completion: {stage2_completion_tokens}"
                else:
                     token_info = " (token data unavailable)" # Fallback if still None
                st.caption(f"*Processing time: {stage2_time:.2f}s{token_info}*")

        # Append the second message to history
        st.session_state.messages.append({"role": "assistant", "content": detailed_response_content})
        # Add the final action type to the log
        st.session_state.actions.append(f"Action: {final_action_type}")
        # Store the processing time and RETRIEVED token counts for the second message
        message_index = len(st.session_state.messages) - 1
        st.session_state.response_times[message_index] = {
            "time": stage2_time,
            "prompt_tokens": stage2_prompt_tokens, # Use the retrieved value
            "completion_tokens": stage2_completion_tokens # Use the retrieved value
        }

def main():
    st.set_page_config(
        page_title="Food Ordering Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS to increase sidebar width by 50%
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            min-width: 31.5rem !important;
            max-width: 31.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    initialize_session_state()
    
    # Make sure we have a field for response times in session state
    if 'response_times' not in st.session_state:
        st.session_state.response_times = {}
    
    # Make sure we have a field for current cart
    if 'current_cart' not in st.session_state:
        st.session_state.current_cart = []
        
    # Make sure we have a cart status field
    if 'cart_status' not in st.session_state:
        st.session_state.cart_status = "OPEN"
    
    # Render sidebar using the component function
    render_sidebar(st.session_state.menu, st.session_state.actions, st.session_state.current_cart)
    
    # Main chat interface title
    st.title("Food Ordering Chatbot")
    
    # Display chat messages using the component function
    display_chat_messages(st.session_state.messages, st.session_state.response_times)
    
    # Chat input - capture prompt
    prompt = st.chat_input("Type your message here...")
    
    # Handle submission if a prompt was entered
    if prompt:
        # Run the asynchronous handling function
        asyncio.run(handle_chat_submission(prompt))
        # Rerun at the end to update the display after handling completes
        st.rerun()

if __name__ == "__main__":
    main() 