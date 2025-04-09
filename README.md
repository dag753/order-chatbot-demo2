# Food Ordering Chatbot

A Streamlit-based food ordering chatbot that uses LlamaIndex workflows for conversation management. The chatbot allows users to inquire about a menu, place orders, and engage in general conversation.

## Features

- Interactive chat interface
- Menu management (add, view, remove items)
- Chat history tracking
- Action logging
- Intelligent conversation routing based on user intent

## Architecture

The application follows an architecture where:
- The main chat interface is on the right side
- Menu management, chat history controls, and action logging are on the left side
- LlamaIndex workflows handle the conversation routing and response generation

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

Run the application with:

```
streamlit run app.py
```

## Usage

1. Add menu items using the form in the sidebar
2. Interact with the chatbot by typing messages in the chat input
3. View the chatbot's actions and decision-making process in the sidebar
4. Clear the chat history using the button in the sidebar when needed

## Technical Details

- Built with Streamlit for the UI
- Uses LlamaIndex for conversation workflows and context management
- Leverages OpenAI's GPT models for natural language understanding
- Implements a router-based architecture for handling different conversation paths 