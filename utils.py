import streamlit as st
import json
import os

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "actions" not in st.session_state:
        st.session_state.actions = []
    
    if "response_times" not in st.session_state:
        st.session_state.response_times = {}
        
    if "menu" not in st.session_state:
        # Initialize with some default menu items
        st.session_state.menu = {
            "Margherita Pizza": {
                "description": "Classic pizza with tomato sauce, mozzarella, and basil",
                "price": 12.99
            },
            "Veggie Burger": {
                "description": "Plant-based patty with lettuce, tomato, and special sauce",
                "price": 9.99
            },
            "Caesar Salad": {
                "description": "Romaine lettuce, croutons, parmesan cheese with Caesar dressing",
                "price": 8.49
            }
        }
    
    if "current_order" not in st.session_state:
        st.session_state.current_order = []

def menu_to_string(menu):
    """Convert menu dictionary to a formatted string."""
    menu_str = "MENU:\n"
    for item, details in menu.items():
        menu_str += f"- {item}: ${details['price']:.2f} - {details['description']}\n"
    return menu_str

def save_menu(menu, filepath="menu.json"):
    """Save menu to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(menu, f, indent=4)

def load_menu(filepath="menu.json"):
    """Load menu from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {} 