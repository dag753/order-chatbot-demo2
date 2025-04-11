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
        # Load menu from file instead of hardcoding
        st.session_state.menu = load_menu("data/menu.json")
        # If the file doesn't exist or is empty, initialize with an empty dict
        if not st.session_state.menu:
             st.session_state.menu = {} # Or provide a minimal default
    
    if "current_cart" not in st.session_state:
        st.session_state.current_cart = []
        
    if "cart_status" not in st.session_state:
        st.session_state.cart_status = "OPEN"

def menu_to_string(menu):
    """Convert menu dictionary (with categories and options) to a formatted string."""
    menu_str = "MENU:\\n"
    for category, items in menu.items():
        menu_str += f"\\n--- {category.upper()} ---\\n"
        if isinstance(items, dict): # Check if it's a dictionary of items
            for item, details in items.items():
                # Main item line
                menu_str += f"- {item}"
                if details.get("price") is not None:
                     menu_str += f": ${details['price']:.2f}"
                if details.get("description"):
                    menu_str += f" - {details['description']}"
                menu_str += "\\n"

                # List options if they exist
                if details.get("options"):
                    menu_str += "    Options:\\n"
                    for option, price_mod in details["options"].items():
                         modifier = f"+${price_mod:.2f}" if price_mod > 0 else "(no charge)" if price_mod == 0 else f"-${abs(price_mod):.2f}"
                         menu_str += f"      - {option.replace('_', ' ').title()}: {modifier}\\n"
        else:
            # Handle cases where category value isn't a dict (e.g., maybe a simple list or string if format changes)
            menu_str += f"  (Category format unexpected for {category})\\n"

    # Add a separate section for general substitutions/add-ons if they exist at the top level
    if "Substitutions" in menu and isinstance(menu["Substitutions"], dict):
        menu_str += f"\\n--- GENERAL ADD-ONS/SUBSTITUTIONS ---\\n"
        for sub, details in menu["Substitutions"].items():
             modifier = f"+${details['price']:.2f}" if details['price'] > 0 else "(no charge)"
             menu_str += f"- {sub.replace('_', ' ').title()}: {modifier}\\n"
    if "Sauces (Extra)" in menu and isinstance(menu["Sauces (Extra)"], dict):
         menu_str += f"\\n--- EXTRA SAUCES ---\\n"
         for sauce, details in menu["Sauces (Extra)"].items():
             modifier = f"+${details['price']:.2f}" if details['price'] > 0 else "(free)"
             menu_str += f"- {sauce}: {modifier}\\n"

    return menu_str

def save_menu(menu, filepath="data/menu.json"):
    """Save menu to a JSON file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(menu, f, indent=4)

def load_menu(filepath="data/menu.json"):
    """Load menu from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {} 