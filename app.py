import os, sys
import logging


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


########################################################
# Configuration
########################################################

# User configuration
USER_NAME = "Alex"
MODEL_NAME = "gpt-4"
MAX_TOKENS = 500

# Streamlit page configuration
st.set_page_config(
    page_title="H&M Dress Recommendations", 
    page_icon="💃", 
    layout="centered", 
    initial_sidebar_state="expanded",
)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        welcome_message = generate_welcome_message()
        st.session_state["messages"] = [
            {"role": "assistant", "content": welcome_message}
        ]
    if "message_counter" not in st.session_state:
        st.session_state["message_counter"] = 0


########################################################
# Prompts
########################################################

def generate_welcome_message():
    global USER_NAME
    
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,  # Use the variable instead of the string
        messages=[
            {"role": "assistant", "content": f"Generate a friendly greeting for a clothing recommendation app. use the name of the user:{USER_NAME} in the greeting. The greeting should introduce an AI assistant ready to help find perfect style recommendations."}
        ],
        max_tokens=50
    )
    return response.choices[0].message['content'].strip()


def ask_for_more_details(history_text: str):
    response = openai.ChatCompletion.create(
        model = MODEL_NAME,
        messages = [
            {"role": "assistant", "content": f"Ask the user for more details about their clothing preferences. Mention that they can specify the type of clothing, color, or any other preferences they have."}
        ],
        max_tokens=30
    )
    return response.choices[0].message['content'].strip()


def ask_to_rephrase_request():
    """Generate a response asking the user to rephrase their question about clothing."""
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "assistant", "content": "Please rephrase your question. You can ask about clothing style, color, size, etc."}
        ],
        max_tokens=30
    )
    return response.choices[0].message['content'].strip()


def ask_for_more_details_and_verify(input_text: str):
    pass


########################################################
# Condition
########################################################

def verify_user_input(input_text: str):
    
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that determines if users are asking about clothing purchases. First line should be 'true' or 'false', second line should be a brief reason why."},
            {"role": "user", "content": f'Is this text asking about buying clothes: "{input_text}"?'}
        ],
        max_tokens=50
    )

    # Split the response into lines and process
    lines = response.choices[0].message['content'].strip().split('\n')
    result = lines[0].lower().strip()
    reason = lines[1].strip() if len(lines) > 1 else "No reason provided"
    
    if result == "true":
        return True, reason
    
    elif result == "false":
        return False, reason
    
    else:
        raise ValueError(f"Invalid response from OpenAI: {result}")
    

def verify_enough_details(input_text: str):
    
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that checks if the user has provided enough details about the clothing they want to buy. First line should be 'true' or 'false', second line should be a brief reason why."},
            {"role": "user", "content": f'Is this text providing enough details about buying clothes: "{input_text}"? Check if it mentions at least two of the following: product name, product type, product group, graphical appearance, color group, perceived color value, perceived color master, department, index, index group, section, garment group, or detail description of the clothing.'}
        ],
        max_tokens=50
    )

    # Split the response into lines and process
    lines = response.choices[0].message['content'].strip().split('\n')
    result = lines[0].lower().strip()
    reason = lines[1].strip() if len(lines) > 1 else "No reason provided"
    
    if result == "true":
        return True, reason
    
    elif result == "false":
        return False, reason
    
    else:
        raise ValueError(f"Invalid response from OpenAI: {result}")


########################################################

def display_chat_history():
    for message in st.session_state.get("messages", []):
        if message["role"] == "user":
            st.markdown(f"**{USER_NAME}:** {message['content']}")

        elif message["role"] == "assistant":
            st.markdown(f"**Fashion Assistant:** {message['content']}")
            # Display image if the message contains image information
            if "image_path" in message:
                st.image(message["image_path"], caption=message.get("caption", ""), use_container_width=True)


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
  
        valid_user_input, valid_user_input_reason = verify_user_input(st.session_state["messages"][-1]["content"])

        if valid_user_input:

            #to do: ask for more details
            # Get AI response
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=st.session_state["messages"],
                max_tokens=MAX_TOKENS
            )

            # Add assistant message to history
            assistant_message = response['choices'][0]['message']['content']
            
            st.session_state["messages"].append({"role": "assistant", "content": assistant_message})



            print(f"Buying clothes. Reason: {valid_user_input_reason}")

            #hisotry of information
            history_of_information = "".join([message["role"] + ": " + message["content"] + "\n" for message in st.session_state["messages"]])
        
            valid_enough_details, valid_enough_details_reason = verify_enough_details(history_of_information)

            if valid_enough_details:
                print(f"Enough details. Reason: {valid_enough_details_reason}")


                search_results = SE.embedding_search(history_of_information, k_top=3).matches

                # Add search results to the chat history
                for result in search_results:
                    image_path = SE.get_image_path(result["id"])

                    # print(f"Image path for ID {result['id']}: {image_path}")  # Debugging output
                    
                    # Instead of displaying the image directly, store it in the message
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Found item with ID: {result['id']}.",
                        "image_path": image_path,
                        "caption": f"Found item with ID: {result['id']}"
                    })        

            else:
                pass
                #to do: ask for more details
        
        else:
            print(f"non relevant user input. Reason: {valid_user_input_reason}")
            assistant_message = ask_to_rephrase_request()

            st.session_state["messages"].append({"role": "assistant", "content": assistant_message})


        # Rerun to refresh the page and show new messages
        st.rerun()

# Run the app
if __name__ == "__main__":
    main()

