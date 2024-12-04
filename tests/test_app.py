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

@pytest.fixture
def conversation():
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
            "content": "I am looking for a white bra for sport.",
        },
    ]

def test_fashion_recommendation(conversation):
    # Send the conversation to the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
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

    assert "sports bra" in assistant_reply.lower() or "sport bra" in assistant_reply.lower(), \
        "The assistant should mention 'sports bra' in the recommendations."

    assert "white" in assistant_reply.lower(), \
        "The assistant should mention 'white' in the recommendations."