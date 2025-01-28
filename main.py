import logging
import streamlit as st
import numpy as np
from langchain_ollama import ChatOllama
from typing import List, Dict
import time
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatbotApp:
    def __init__(self):
        """Initialize the chatbot application with configuration and state management."""
        self.initialize_session_state()
        self.setup_llm()

    def initialize_session_state(self):
        """Initialize or reset session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'model_config' not in st.session_state:
            st.session_state.model_config = {
                'model_name': 'llama3.2',
                'temperature': 0.7,
                'chunk_size': 500
            }

    def setup_llm(self):
        """Set up the language model with current configuration."""
        try:
            self.llm = ChatOllama(
                model=st.session_state.model_config['model_name'],
                temperature=st.session_state.model_config['temperature'],
            )
            logger.info(f"LLM initialized with model: {st.session_state.model_config['model_name']}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            st.error("Failed to initialize the language model. Please check your configuration.")

    def extract_response_content(self, response) -> str:
        """
        Extract the actual content from the LLM response object.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            str: The cleaned response content
        """
        try:
            # If response is already a string, return it
            if isinstance(response, str):
                return response
            
            # If response has a content attribute, extract it
            if hasattr(response, 'content'):
                content = response.content
                # Remove any think tags if present
                content = content.replace('<think>\n\n</think>\n\n', '')
                return content
                
            # If response is a dictionary, try to get content
            if isinstance(response, dict) and 'content' in response:
                return response['content']
            
            # If we can't extract content, convert the whole response to string
            return str(response)
            
        except Exception as e:
            logger.error(f"Error extracting response content: {e}")
            return "Sorry, I had trouble processing the response."

    def generate_response(self, input_text: str) -> str:
        """
        Generate a response from the model with error handling and retry logic.
        
        Args:
            input_text (str): The user's input text
            
        Returns:
            str: The model's response
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with st.spinner('Generating response...'):
                    raw_response = self.llm.invoke(input_text)
                    response = self.extract_response_content(raw_response)
                    logger.info("Successfully generated response")
                    return response
            except Exception as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count} failed: {e}")
                if retry_count == max_retries:
                    logger.error(f"Failed to generate response after {max_retries} attempts")
                    return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."
                time.sleep(1)  # Short delay before retry

    def create_ui(self):
        """Create and configure the Streamlit user interface."""
        st.set_page_config(
            page_title="Advanced AI Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply custom CSS styling
        st.markdown("""
            <style>
            .stButton>button {
                width: 100%;
                margin-top: 1rem;
            }
            .chat-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            .user-message {
                background-color: #e9ecef;
            }
            .assistant-message {
                background-color: #f8f9fa;
            }
            .metadata {
                font-size: 0.8rem;
                color: #6c757d;
            }
            </style>
            """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            model_name = st.selectbox(
                "Model",
                ["llama3.2", "deepseek-r1:7b", "mistral"],
                index=["llama3.2", "deepseek-r1:7b", "mistral"].index(
                    st.session_state.model_config['model_name']
                )
            )
            
            # Model parameters
            temperature = st.slider(
                "Temperature",
                0.0, 1.0,
                st.session_state.model_config['temperature']
            )
            
            chunk_size = st.number_input(
                "Chunk Size",
                100, 1000,
                st.session_state.model_config['chunk_size']
            )
            
            # Update configuration if changed
            if (model_name != st.session_state.model_config['model_name'] or
                temperature != st.session_state.model_config['temperature'] or
                chunk_size != st.session_state.model_config['chunk_size']):
                
                st.session_state.model_config.update({
                    'model_name': model_name,
                    'temperature': temperature,
                    'chunk_size': chunk_size
                })
                self.setup_llm()
                st.success("Configuration updated!")
            
            # Add debug mode toggle
            if st.checkbox("Debug Mode", False):
                st.session_state.debug_mode = True
            else:
                st.session_state.debug_mode = False
            
            # Clear chat history button
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.success("Chat history cleared!")

    def display_chat_history(self):
        """Display the chat history with enhanced formatting."""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if st.session_state.get('debug_mode', False):
                    st.markdown(f"<p class='metadata'>Time: {msg['timestamp']}</p>",
                              unsafe_allow_html=True)
                    if 'metadata' in msg:
                        st.json(msg['metadata'])

    def main(self):
        """Main application logic and UI rendering."""
        self.create_ui()
        
        # Title and description
        st.title("Advanced AI Chatbot")
        st.markdown("### An intelligent chatbot powered by state-of-the-art language models")
        
        # Render sidebar
        self.render_sidebar()
        
        # Display current model info
        st.write(f"Current model: **{st.session_state.model_config['model_name']}**")
        
        # Display chat history
        self.display_chat_history()
        
        # User input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Generate and display response
            raw_response = self.generate_response(user_input)
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": raw_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "model": st.session_state.model_config['model_name'],
                    "temperature": st.session_state.model_config['temperature']
                }
            })
            
            # Rerun to update chat display
            st.experimental_rerun()

if __name__ == "__main__":
    app = ChatbotApp()
    app.main()


# import logging
# import streamlit as st
# import numpy as np
# from langchain_ollama import ChatOllama

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize ChatOllama
# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0,
# )

# def generate_response(input_text):
#     """Generate a response from the model."""
#     try:
#         response = llm.invoke(input_text)
#         st.info(response)
#     except Exception as e:
#         logger.error(f"Error generating response: {e}")
#         st.error("An error occurred while generating the response.")

# def create_streamlit_ui():
#     """Set up the Streamlit UI with enhanced features."""
#     st.set_page_config(
#         page_title="Deepseek AI Document Q&A",
#         page_icon="🤖",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     # Custom CSS for styling
#     st.markdown(
#         """
#         <style>
#         .stButton>button {
#             width: 100%;
#             margin-top: 1rem;
#         }
#         .success-message {
#             padding: 1rem;
#             background-color: #d4edda;
#             border-color: #c3e6cb;
#             color: #155724;
#             border-radius: 0.25rem;
#             margin-bottom: 1rem;
#         }
#         </style>
#         """, 
#         unsafe_allow_html=True
#     )

#     # Initialize session state variables
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'rag_app' not in st.session_state:
#         st.session_state.rag_app = None

#     return st.session_state

# def main():
#     """Main application logic."""
#     session_state = create_streamlit_ui()

#     # Title and description
#     st.title("Deepseek & LLAMA Chatbot")
#     st.markdown("### This is a chatbot that can answer questions using Deepseek.")

#     # Sidebar configuration
#     with st.sidebar:
#         st.header("⚙️ Configuration")
#         model_name = st.selectbox(
#             "Model", 
#             ["llama3.2", "deepseek-r1:7b", "mistral"], 
#             index=0
#         )
#         temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
#         chunk_size = st.number_input("Chunk Size", 100, 1000, 500)

#     st.write(f'You are currently working with {model_name}')

#     # User input for chatbot prompt
#     user_input = st.text_input("Enter your prompt:")

#     if user_input:
#         with st.chat_message("user"):
#             st.write(user_input)
#         with st.chat_message("assistant"):
#             generate_response(user_input)

# if __name__ == "__main__":
#     main()
