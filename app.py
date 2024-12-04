import streamlit as st
import openai


from user_info import OPENAI_API_KEY


# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY


# Streamlit app
st.title("Chatbot with OpenAI")


# Sidebar for settings
st.sidebar.title("Settings")
model = st.sidebar.selectbox("Choose a model", ["gpt-3.5-turbo", "gpt-4"])
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150)


# Initialize chat history
if "messages" not in st.session_state:
   st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]


# User input
user_input = st.text_input("You:", key="user_input")


if st.button("Send"):
   if user_input:
       # Add user message to chat history
       st.session_state["messages"].append({"role": "user", "content": user_input})


       # Get response from OpenAI
       response = openai.ChatCompletion.create(
           model=model,
           messages=st.session_state["messages"],
           max_tokens=max_tokens
       )


       # Extract the assistant's message
       assistant_message = response['choices'][0]['message']['content']


       # Add assistant message to chat history
       st.session_state["messages"].append({"role": "assistant", "content": assistant_message})


# Display chat history
for message in st.session_state["messages"]:
   if message["role"] == "user":
       st.markdown(f"**You:** {message['content']}")
   elif message["role"] == "assistant":
       st.markdown(f"**Chatbot:** {message['content']}")
