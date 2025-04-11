import streamlit as st
from typing import List, Dict, Any
import logging
import re

def display_cart(cart_items: List[Dict[str, Any]]):
    """Displays the shopping cart in the top right corner of the UI."""
    cart_status = st.session_state.get("cart_status", "OPEN")
    
    status_colors = {
        "OPEN": "#4A4A4A",
        "PENDING CONFIRMATION": "#FFA500",
        "CONFIRMED": "#00AA00"
    }
    
    status_color = status_colors.get(cart_status, "#4A4A4A")
    
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
        st.sidebar.info("Empty Cart")
        return
    
    cart_data = []
    total = 0.0
    
    for item in cart_items:
        if not isinstance(item, dict):
            logging.warning(f"Invalid cart item type: {type(item)} - {item}")
            continue
            
        item_name = item.get("item", "Unknown item")
        quantity = item.get("quantity", 1)
        price = item.get("price", 0.0)
        options = ", ".join(item.get("options", []) if isinstance(item.get("options"), list) else [])
        
        item_total = quantity * price
        total += item_total
        
        cart_data.append({
            "Item": f"{item_name}{' with ' + options if options else ''}",
            "Qty": quantity,
            "Price": f"${price:.2f}",
            "Total": f"${item_total:.2f}"
        })
    
    if cart_data:
        st.sidebar.dataframe(cart_data, hide_index=True, use_container_width=True)
        
        st.sidebar.markdown(f"<div style='text-align: right; font-weight: bold; color: {status_color};'>Total: ${total:.2f}</div>", 
                   unsafe_allow_html=True)

def render_sidebar(menu: Dict[str, Dict[str, Any]], actions: List[str], cart_items: List[Dict[str, Any]]):
    """Renders the sidebar UI elements for menu management and action logs."""
    display_cart(cart_items)
    st.sidebar.divider()
    
    st.sidebar.title("Chatbot Info & Control")

    st.sidebar.subheader("Chatbot Actions")
    if actions:
        for action in reversed(actions):
            st.sidebar.info(action)
    else:
        st.sidebar.write("No actions yet.")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.actions = []
        st.session_state.response_times = {}
        st.session_state.current_cart = []
        st.session_state.cart_status = "OPEN"
        st.rerun()

    st.sidebar.divider()

    st.sidebar.title("Menu Management")

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
            st.rerun()

    st.sidebar.subheader("Current Menu")
    if menu:
        for category, items in menu.items():
            if isinstance(items, dict):
                st.sidebar.markdown(f"**__{category}__**")
                category_items = list(items.items())
                for item, details in category_items:
                    if isinstance(details, dict):
                        price_str = f" - ${details.get('price', 0.0):.2f}" if details.get('price') is not None else ""
                        st.sidebar.write(f"**{item}**{price_str}")

                        description = details.get('description')
                        options = details.get('options')

                        if description or options:
                            with st.sidebar.expander(f"Details for {item}"):
                                if description:
                                    st.write(f"{description}")
                                if options and isinstance(options, dict):
                                    st.write("Options:")
                                    for opt_name, opt_price in options.items():
                                        mod = f"+${opt_price:.2f}" if opt_price > 0 else "(no charge)" if opt_price == 0 else f"-${abs(opt_price):.2f}"
                                        st.write(f"  - {opt_name.replace('_', ' ').title()}: {mod}")

                        if st.sidebar.button(f"Remove {item}", key=f"remove_{category}_{item}"):
                             if category in st.session_state.menu and item in st.session_state.menu[category]:
                                 del st.session_state.menu[category][item]
                                 if not st.session_state.menu[category]:
                                     del st.session_state.menu[category]
                                 st.sidebar.success(f"Removed {item} from {category}")
                                 st.rerun()
                             else:
                                 st.sidebar.error(f"Could not remove {item} from {category}. Item or category not found.")
                    else:
                         st.sidebar.write(f"- {item}: (Details format incorrect)")
            else:
                 st.sidebar.write(f"- {category}: (Unexpected top-level item format)")
    else:
        st.sidebar.write("No items in menu. Add some above!")

def display_chat_messages(messages: List[Dict[str, str]], response_times: Dict[int, Dict[str, Any]]):
    """Displays the chat history in the main area."""
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            st.text(content)

            if message["role"] == "assistant" and i in response_times:
                response_info = response_times[i]
                
                if isinstance(response_info, float):
                    st.caption(f"*Response time: {response_info:.2f}s*")
                else:
                    time_taken = response_info.get("time", 0)
                    prompt_tokens = response_info.get("prompt_tokens")
                    completion_tokens = response_info.get("completion_tokens")
                    
                    if time_taken > 0:
                        caption = f"*Response time: {time_taken:.2f}s"
                        if prompt_tokens is not None:
                            caption += f", prompt tokens: {prompt_tokens}"
                        if completion_tokens is not None:
                            caption += f", completion tokens: {completion_tokens}"
                        if prompt_tokens is None and completion_tokens is None:
                            caption += " (token data unavailable)"
                        caption += "*"
                        st.caption(caption) 