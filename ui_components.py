import streamlit as st
from typing import List, Dict, Any
import logging
import re  # Add regex import for text fixing

def display_cart(cart_items: List[Dict[str, Any]]):
    """Displays the shopping cart in the top right corner of the UI."""
    # Get cart status from session state
    cart_status = st.session_state.get("cart_status", "OPEN")
    
    # Define status colors
    status_colors = {
        "OPEN": "#4A4A4A",  # Default gray
        "PENDING CONFIRMATION": "#FFA500",  # Orange
        "CONFIRMED": "#00AA00"  # Green
    }
    
    status_color = status_colors.get(cart_status, "#4A4A4A")
    
    # Use markdown to style the cart header with status
    st.sidebar.markdown(f"""
    <style>
    .cart-header {{
        text-align: right;
        color: {status_color};
        padding: 5px;
        font-weight: bold;
    }}
    .cart-status {{
        color: {status_color};
        font-weight: bold;
    }}
    </style>
    <div class="cart-header">Your Order</div>
    <div class="cart-status">Status: {cart_status}</div>
    """, unsafe_allow_html=True)
    
    if not cart_items:
        # Display "Empty Cart" message when cart is empty
        st.sidebar.info("Empty Cart")
        return
    
    # Create a dataframe to display the cart items
    cart_data = []
    total = 0.0
    
    for item in cart_items:
        # Add type checking to handle case where item might be a string
        if not isinstance(item, dict):
            # Log error and skip this item
            logging.warning(f"Invalid cart item type: {type(item)} - {item}")
            continue
            
        item_name = item.get("item", "Unknown item")
        quantity = item.get("quantity", 1)
        price = item.get("price", 0.0)
        options = ", ".join(item.get("options", []) if isinstance(item.get("options"), list) else [])
        
        # Calculate item total
        item_total = quantity * price
        total += item_total
        
        # Add to cart data
        cart_data.append({
            "Item": f"{item_name}{' with ' + options if options else ''}",
            "Qty": quantity,
            "Price": f"${price:.2f}",
            "Total": f"${item_total:.2f}"
        })
    
    # Display cart items as a table
    if cart_data:
        st.sidebar.dataframe(cart_data, hide_index=True, use_container_width=True)
        
        # Display total
        st.sidebar.markdown(f"<div style='text-align: right; font-weight: bold; color: {status_color};'>Total: ${total:.2f}</div>", 
                   unsafe_allow_html=True)

def render_sidebar(menu: Dict[str, Dict[str, Any]], actions: List[str], cart_items: List[Dict[str, Any]]):
    """Renders the sidebar UI elements for menu management and action logs."""
    # --- Display Cart at the very top --- 
    display_cart(cart_items)
    st.sidebar.divider() # Add divider after cart
    
    st.sidebar.title("Chatbot Info & Control") # General Title

    # Chat actions/logs display first
    st.sidebar.subheader("Chatbot Actions")
    if actions:
        for action in reversed(actions): # Show newest first
            st.sidebar.info(action)
    else:
        st.sidebar.write("No actions yet.")

    # Clear chat history button (Placed after actions for context)
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.actions = []
        st.session_state.response_times = {}
        st.rerun()

    st.sidebar.divider() # Add a visual separator

    # Then Menu Management
    st.sidebar.title("Menu Management")

    # Menu input form
    # Note: Modifying session state directly here based on form submission
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
            st.rerun() # Rerun after adding

    # Display current menu
    st.sidebar.subheader("Current Menu")
    if menu:
        # Iterate through categories first
        for category, items in menu.items():
            # Check if items is a dictionary (expected category structure)
            if isinstance(items, dict):
                st.sidebar.markdown(f"**__{category}__**") # Display category name
                # Iterate through items within the category
                category_items = list(items.items()) # Copy for safe iteration if needed
                for item, details in category_items:
                    # Ensure details is a dictionary before trying to access keys
                    if isinstance(details, dict):
                        price_str = f" - ${details.get('price', 0.0):.2f}" if details.get('price') is not None else ""
                        st.sidebar.write(f"**{item}**{price_str}")

                        description = details.get('description')
                        options = details.get('options')

                        # Use expander only if there's description or options
                        if description or options:
                            with st.sidebar.expander(f"Details for {item}"):
                                if description:
                                    st.write(f"{description}")
                                if options and isinstance(options, dict):
                                    st.write("Options:")
                                    for opt_name, opt_price in options.items():
                                        mod = f"+${opt_price:.2f}" if opt_price > 0 else "(no charge)" if opt_price == 0 else f"-${abs(opt_price):.2f}"
                                        st.write(f"  - {opt_name.replace('_', ' ').title()}: {mod}")

                        # --- Updated Remove button logic ---
                        # The key uniquely identifies the item within its category
                        if st.sidebar.button(f"Remove {item}", key=f"remove_{category}_{item}"):
                             if category in st.session_state.menu and item in st.session_state.menu[category]:
                                 del st.session_state.menu[category][item]
                                 # Optional: Remove the category if it's now empty
                                 if not st.session_state.menu[category]:
                                     del st.session_state.menu[category]
                                 st.sidebar.success(f"Removed {item} from {category}")
                                 st.rerun()
                             else:
                                 st.sidebar.error(f"Could not remove {item} from {category}. Item or category not found.")
                    else:
                         st.sidebar.write(f"- {item}: (Details format incorrect)")
            else:
                # Handle cases where top-level item isn't a category dictionary
                 st.sidebar.write(f"- {category}: (Unexpected top-level item format)")

    else:
        st.sidebar.write("No items in menu. Add some above!")

    # Chat actions/logs display - REMOVED FROM HERE
    # st.sidebar.subheader("Chatbot Actions")
    # if actions:
    #     for action in reversed(actions): # Show newest first
    #         st.sidebar.info(action)
    # else:
    #     st.sidebar.write("No actions yet.")


def display_chat_messages(messages: List[Dict[str, str]], response_times: Dict[int, Dict[str, Any]]):
    """Displays the chat history in the main area."""
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            st.markdown(content, unsafe_allow_html=False)
            
            # Display response time and token counts if available for assistant messages
            if message["role"] == "assistant" and i in response_times:
                # Get response info - handle both the old format (float) and new format (dict)
                response_info = response_times[i]
                
                if isinstance(response_info, float):
                    # Handle old format (just a time value)
                    st.caption(f"*Response time: {response_info:.2f}s*")
                else:
                    # Handle new format (dictionary with time and token counts)
                    time_taken = response_info.get("time", 0)
                    prompt_tokens = response_info.get("prompt_tokens")
                    completion_tokens = response_info.get("completion_tokens")
                    
                    # Basic debugging - show raw token data for verification
                    if time_taken > 0:
                        caption = f"*Response time: {time_taken:.2f}s"
                        # Add token counts if available
                        if prompt_tokens is not None:
                            caption += f", prompt tokens: {prompt_tokens}"
                        if completion_tokens is not None:
                            caption += f", completion tokens: {completion_tokens}"
                        if prompt_tokens is None and completion_tokens is None:
                            caption += " (token data unavailable)"
                        caption += "*"
                        st.caption(caption)
                        
                        # For debugging - show the raw response_info
                        # token_debug = f"Debug - Raw response info: {response_info}"
                        # st.caption(token_debug) 