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

# Set up logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("food_ordering_bot")
logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()

async def process_message(workflow, user_query):
    """Process a message through the workflow and return the response"""
    try:
        # Measure total workflow time
        total_start_time = time.time()
        
        # Create a start event
        start_event = StartEvent(content=user_query)
        logger.info("Debug: Created StartEvent")
        
        # Attempt to run workflow (note: timeout is set when creating the workflow)
        logger.info("Debug: About to run workflow")
        # Run without timeout parameter since it's not supported with start_event
        result = await workflow.run(start_event=start_event)
        
        # Calculate and log total workflow time
        total_elapsed = time.time() - total_start_time
        logger.info(f"==== Total workflow processing time: {total_elapsed:.2f}s ====")
        
        # Log the result type and str representation
        result_type = type(result).__name__ if result is not None else "None"
        logger.info(f"Debug: Workflow result type: {result_type}")
        
        # Check result validity more carefully
        if result is None:
            logger.error("Workflow returned None unexpectedly")
            # Create a fallback StopEvent
            return ChatResponseStopEvent(
                result=None,
                response="I'm sorry, but something went wrong with my processing. The workflow returned no response.",
                action_type="error"
            )
            
        # Check if the result is our custom ChatResponseStopEvent
        if isinstance(result, ChatResponseStopEvent):
            logger.info(f"Debug: Got ChatResponseStopEvent with response: {result.response[:30]}...")
            logger.info(f"Debug: Action type: {result.action_type}")
            return result
        elif isinstance(result, StopEvent):
            # Handle case where we get a regular StopEvent (shouldn't happen)
            logger.warning(f"Debug: Got regular StopEvent instead of ChatResponseStopEvent")
            return ChatResponseStopEvent(
                result=None,
                response=str(result.result) if result.result else "No response available",
                action_type="unknown"
            )
        else:
            # Unexpected result type
            logger.error(f"Debug: Unexpected result type: {result_type}")
            return ChatResponseStopEvent(
                result=None,
                response=f"Got unexpected result type: {result_type}",
                action_type="error"
            )
        
    except Exception as e:
        # Log the full exception
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        # Include stack trace for debugging
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        # Make sure we return something valid here
        return ChatResponseStopEvent(
            result=None,
            response=f"I'm sorry, I couldn't process your request. Error: {str(e)}",
            action_type="error"
        )

def main():
    st.set_page_config(
        page_title="Food Ordering Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    initialize_session_state()
    
    # Make sure we have a field for response times in session state
    if 'response_times' not in st.session_state:
        st.session_state.response_times = {}
    
    # Sidebar: Menu Management
    st.sidebar.title("Menu Management")
    
    # Menu input form
    with st.sidebar.form("menu_form"):
        st.write("Add Menu Item")
        item_name = st.text_input("Item Name")
        item_description = st.text_input("Description")
        item_price = st.number_input("Price", min_value=0.0, step=0.01)
        add_button = st.form_submit_button("Add Item")
        
    if add_button and item_name and item_price > 0:
        st.session_state.menu[item_name] = {
            "description": item_description,
            "price": item_price
        }
        st.sidebar.success(f"Added {item_name} to menu!")
    
    # Display current menu
    st.sidebar.subheader("Current Menu")
    if st.session_state.menu:
        for item, details in st.session_state.menu.items():
            st.sidebar.write(f"**{item}** - ${details['price']:.2f}")
            st.sidebar.write(f"_{details['description']}_")
            if st.sidebar.button(f"Remove {item}"):
                del st.session_state.menu[item]
                st.rerun()
    else:
        st.sidebar.write("No items in menu. Add some above!")
    
    # Clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.actions = []
        st.rerun()
    
    # Chat actions/logs
    st.sidebar.subheader("Chatbot Actions")
    for action in st.session_state.actions:
        st.sidebar.info(action)
    
    # Main chat interface on the right
    st.title("Food Ordering Chatbot")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Show the message content
            st.markdown(message["content"], unsafe_allow_html=False)
            
            # If this is an assistant message and it has a corresponding response time, show it
            if message["role"] == "assistant" and i in st.session_state.response_times:
                time_taken = st.session_state.response_times[i]
                st.caption(f"*Response time: {time_taken:.2f}s*")

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        
        # Add a log of the user's message
        st.session_state.actions.append(f"User said: {prompt}")
        
        # Generate response
        chat_message_placeholder = st.chat_message("assistant")
        
        # Create workflow using the chat engine with history context
        chat_workflow = create_chat_engine(
            st.session_state.menu, 
            st.session_state.messages[:-1]  # exclude the most recent user message
        )
        
        # Process the message through the workflow
        logger.info(f"Processing user message: {prompt[:30]}...")
        
        # Start timing the full processing
        full_processing_start = time.time()
        
        # Run the workflow processing
        response = asyncio.run(process_message(chat_workflow, prompt))
        
        # Calculate total end-to-end processing time (Stage 1)
        full_processing_time = time.time() - full_processing_start
        logger.info(f"====== TOTAL END-TO-END PROCESSING TIME (Stage 1): {full_processing_time:.2f}s ======")
        
        logger.info(f"Response received: {response}")
        
        # Handle the response
        final_response_displayed = False # Flag to track if a response was displayed
        try:
            if response is None:
                error_message = "Sorry, I couldn't process your request right now. Please try again later."
                with chat_message_placeholder:
                     st.markdown(error_message, unsafe_allow_html=False)
                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                # Store the response time for the error message
                st.session_state.response_times[len(st.session_state.messages) - 1] = full_processing_time
                st.session_state.actions.append("Action: error")
                final_response_displayed = True
            else:
                # Get the initial response (could be ack or final)
                display_text = response.response
                logger.info(f"Initial response (first 50 chars): {display_text[:50]}...")
                
                # Check if display_text is empty or just a fragment
                if not display_text or display_text.strip() in ['{', '}', '[]', '[', ']', '{}']:
                    display_text = "I'm sorry, I didn't generate a proper response. Please try again."
                    logger.warning(f"Found empty or invalid fragment: {display_text}")
                
                # Display the initial response in the first placeholder
                with chat_message_placeholder:
                     st.markdown(f"{display_text}", unsafe_allow_html=False)
                final_response_displayed = True # At least the first part displayed
                
                # Log the initial workflow action
                action_type = "unknown" # Default action type
                if hasattr(response, 'action_type'):
                    action_type = response.action_type
                    logger.info(f"Initial Action type: {action_type}")
                    # Log the initial action (might be pending)
                    st.session_state.actions.append(f"Action: {action_type}") 
                
                # Add initial assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": display_text})
                # Store the response time for this message
                st.session_state.response_times[len(st.session_state.messages) - 1] = full_processing_time
                
                # --- Stage 2: Handle Pending Actions --- 
                if action_type in ["menu_inquiry_pending", "order_action_pending"]:
                    logger.info(f"Handling pending action: {action_type}")
                    # Use a placeholder for the second message + spinner
                    stage2_placeholder = st.chat_message("assistant")
                    with stage2_placeholder:
                        with st.spinner("Getting details..."): 
                            stage2_start_time = time.time()
                            detailed_response = ""
                            final_action_type = "error"
                            
                            try:
                                original_prompt = prompt # Use the initial user prompt for handlers
                                if action_type == "menu_inquiry_pending":
                                    detailed_response = asyncio.run(chat_workflow._handle_menu_query(original_prompt))
                                    final_action_type = "menu_inquiry"
                                elif action_type == "order_action_pending":
                                    detailed_response = asyncio.run(chat_workflow._handle_order_query(original_prompt))
                                    final_action_type = "order_action"
                                    
                                # Clean the detailed response (similar to initial response handling)
                                if isinstance(detailed_response, str) and \
                                   (detailed_response.strip().startswith('{') and detailed_response.strip().endswith('}')):
                                    try:
                                        json_data = json.loads(detailed_response)
                                        if isinstance(json_data, dict) and 'response' in json_data:
                                            detailed_response = json_data['response']
                                    except json.JSONDecodeError:
                                        pass # Use as is if not valid JSON
                                        
                                stage2_time = time.time() - stage2_start_time
                                logger.info(f"Stage 2 processing time: {stage2_time:.2f}s")

                            except Exception as e:
                                logger.error(f"Error in stage 2 handling ({action_type}): {e}")
                                detailed_response = "Sorry, I encountered an error while getting the details." # Set error msg
                                final_action_type = "error"

                        # Display the detailed response (or error) in the second placeholder
                        st.markdown(detailed_response, unsafe_allow_html=False)
                        
                        # Append the second message to history
                        st.session_state.messages.append({"role": "assistant", "content": detailed_response})
                        # Add the final action type to the log
                        st.session_state.actions.append(f"Action: {final_action_type}")
                        # Store the processing time for the second message using its index
                        st.session_state.response_times[len(st.session_state.messages) - 1] = stage2_time

                    # Rerun at the end to update the full chat display and sidebar
                    st.rerun()
                # --- End Stage 2 --- 
                        
        except Exception as e:
            # Handle any issues accessing response attributes or other errors during display
            logger.error(f"Error processing/displaying response: {e}")
            # If no response was displayed yet, display error in the initial placeholder
            if not final_response_displayed:
                error_message = "Sorry, there was a problem displaying the response."
                with chat_message_placeholder:
                    st.markdown(error_message, unsafe_allow_html=False)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.session_state.actions.append("Action: error")
        
        # Auto-scroll to bottom if not handling stage 2 which does its own rerun
        if not (response and hasattr(response, 'action_type') and response.action_type in ["menu_inquiry_pending", "order_action_pending"]):
             st.rerun()

if __name__ == "__main__":
    main() 