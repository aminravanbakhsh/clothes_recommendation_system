import os, sys

########################################################
# add src directory to the system path
########################################################

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "data")

sys.path.append(os.path.join(root_dir, "src"))

########################################################
# imports
########################################################

from search_engine import SearchEngine
import streamlit as st
import openai
import json
from datetime import datetime
import uuid

from langchain_openai import ChatOpenAI

import pdb

from user_info import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

SE = SearchEngine(data_dir=data_dir, init_vector_database=False)

# User configuration
USER_NAME = "Alex"
MODEL_NAME = "gpt-4"
MAX_TOKENS = 500

# Streamlit page configuration
st.set_page_config(
    page_title="H&M Dress Recommendations", 
    page_icon="💃", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        welcome_message = generate_welcome_message()
        st.session_state["messages"] = [
            {"role": "system", "content": welcome_message}
        ]
    if "message_counter" not in st.session_state:
        st.session_state["message_counter"] = 0

# Generate welcome message
def generate_welcome_message():
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Generate a friendly greeting for a clothing recommendation app. The greeting should introduce an AI assistant ready to help find perfect style recommendations."}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'].strip()

# Display chat history
def display_chat_history():
    for message in st.session_state.get("messages", []):
        if message["role"] == "user":
            st.markdown(f"**{USER_NAME}:** {message['content']}")
        elif message["role"] == "system" or message["role"] == "assistant":
            st.markdown(f"**Chatbot:** {message['content']}")

# Main app
def main():
    # Initialize session state
    init_session_state()

    # Title
    st.html("""<h2 style="text-align: center;">👜👠<i> H&M Personalized Dress and Clothes Recommendations </i> 👗👚</h2>""")

    # Display chat history
    display_chat_history()

    # Input at the bottom of the page
    st.markdown("---")
    with st.form(key='user_input_form', clear_on_submit=True):
        user_input = st.text_input(f"{USER_NAME}:", key="user_input_form_input")
        submit_button = st.form_submit_button(label='Submit')

    # Process user input
    if submit_button and user_input:
        # Increment message counter
        st.session_state["message_counter"] += 1

        # Add user message to history
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Use the SearchEngine to search based on the user input
        search_results = SE.embedding_search(user_input, k_top=3).matches

        # Add search results to the chat history
        for result in search_results:
            image_path = SE.get_image_path(result["id"])
            print(f"Image path for ID {result['id']}: {image_path}")  # Debugging output
            
            # Display the image with its caption using the new parameter
            st.image(image_path, caption=f"Found item with ID: {result['id']}", use_container_width=True)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"Found item with ID: {result['id']}."
            })

        # Get AI response
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=st.session_state["messages"],
            max_tokens=MAX_TOKENS
        )

        # Add assistant message to history
        assistant_message = response['choices'][0]['message']['content']
        
        st.session_state["messages"].append({"role": "assistant", "content": assistant_message})

        # Rerun to refresh the page and show new messages
        st.rerun()

# Run the app
if __name__ == "__main__":
    main()

