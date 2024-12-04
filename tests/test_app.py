import pytest
import os
import sys
import pandas as pd

import openai

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, "src")

sys.path.append(root_dir)
sys.path.append(src_dir)

from search_engine import SearchEngine
from unittest.mock import Mock, patch


# MODEL_NAME = "gpt-4"
MODEL_NAME = "gpt-3.5-turbo"

########################################################

@pytest.fixture
def conversation():
    return [
        {
            "role": "assistant",
            "content": (
                "You are Fashion Assistant, an AI specialized in fashion recommendations. "
                "Help users find clothing items based on their requests."
            ),
        },
        {
            "role": "user",
            "content": "I am looking for a white bra for sport.",
        },
    ]

def test_fashion_recommendation(conversation):
    # Send the conversation to the OpenAI API
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=conversation,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message['content']

    print("\nAssistant's Reply:\n", assistant_reply)

    # Updated assertions with more flexible recommendation phrases
    assert any(phrase in assistant_reply.lower() for phrase in [
        "here are",
        "i recommend",
        "you may consider",
        "options",
        "suggestions"
    ]), "The assistant should provide recommendations."

    assert "sports bra" in assistant_reply.lower() or "sport bra" in assistant_reply.lower(), \
        "The assistant should mention 'sports bra' in the recommendations."

    assert "white" in assistant_reply.lower(), \
        "The assistant should mention 'white' in the recommendations."
    

########################################################


@pytest.fixture
def conversation_red_dress():
    return [
        {
            "role": "system",
            "content": (
                "You are Fashion Assistant, an AI specialized in fashion recommendations. "
                "Help users find clothing items based on their requests."
            ),
        },
        {
            "role": "user",
            "content": "I'm searching for a red evening dress for a gala event.",
        },
    ]

def test_red_evening_dress(conversation_red_dress):
    # Send the conversation to the OpenAI API
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=conversation_red_dress,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message['content']

    print("\nAssistant's Reply:\n", assistant_reply)

    # Assertions to check if the assistant provides recommendations
    assert "Here are some" in assistant_reply or "I recommend" in assistant_reply, \
        "The assistant should provide recommendations."

    assert "evening dress" in assistant_reply.lower() or "gown" in assistant_reply.lower(), \
        "The assistant should mention 'evening dress' or 'gown' in the recommendations."

    assert "red" in assistant_reply.lower(), \
        "The assistant should mention 'red' in the recommendations."

    assert "gala" in assistant_reply.lower() or "formal event" in assistant_reply.lower(), \
        "The assistant should acknowledge that the dress is for a gala or formal event."


########################################################


@pytest.fixture
def conversation_mens_shoes():
    return [
        {
            "role": "system",
            "content": (
                "You are Fashion Assistant, an AI specialized in fashion recommendations. "
                "Help users find clothing items based on their requests."
            ),
        },
        {
            "role": "user",
            "content": "I need a pair of men's black formal shoes for a wedding.",
        },
    ]

def test_mens_black_formal_shoes(conversation_mens_shoes):
    # Send the conversation to the OpenAI API
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=conversation_mens_shoes,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message['content']

    print("\nAssistant's Reply:\n", assistant_reply)

    # Assertions to check if the assistant provides recommendations
    assert "Here are some" in assistant_reply or "I recommend" in assistant_reply, \
        "The assistant should provide recommendations."

    assert any(term in assistant_reply.lower() for term in ["formal shoes", "dress shoes", "oxford"]), \
        "The assistant should mention 'formal shoes', 'dress shoes', or 'oxford' in the recommendations."

    assert "black" in assistant_reply.lower(), \
        "The assistant should mention 'black' in the recommendations."

    assert "men" in assistant_reply.lower() or "men's" in assistant_reply.lower(), \
        "The assistant should specify that the shoes are for men."
    

if __name__ == "__main__":
    test_fashion_recommendation(conversation())
    test_red_evening_dress(conversation_red_dress())
    test_mens_black_formal_shoes(conversation_mens_shoes())
    