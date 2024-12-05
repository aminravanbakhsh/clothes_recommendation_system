# Clothes Recommendation System

A sophisticated AI-powered fashion recommendation system that helps users discover their perfect style.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Conda package manager
- OpenAI account
- Pinecone account
- Kaggle account
- Docker (optional)


### Environment Setup

1. Create a new Conda environment:
```bash
conda create -n proactive python=3.8 -y
conda activate proactive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Create a `user_info.py` in the root directory with the following OpenAI and Pinecone credentials:
```python
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_INDEX_NAME = "your_index_name"
PINECONE_API_KEY = "your_api_key"
PINECONE_ENVIRONMENT = "your_environment"
PINECONE_CLOUD = "your_cloud"
PINECONE_HOST = "your_host"
```

### Data Setup

1. Create a `data` directory in the project root (automatically ignored by Git)
2. Download the dataset from [H&M Fashion Recommendations Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
3. Extract the contents to match this structure:
```
data/
├── images/
└── articles.csv
```

## Usage

### Running the Application
```bash
streamlit run app.py
```

### Managing Streamlit Ports
To free up occupied Streamlit ports:
```bash
ps aux | grep streamlit | grep $(whoami) | awk '{print $2}' | xargs kill -9
```

## Testing

Add your tests to the `tests` directory.

Run the test suite using pytest:
```bash
pytest tests/test_search_engine.py
pytest tests/test_app.py
```

## Docker Support

Build the Docker image:
```bash
docker build --no-cache -t clothes_recommendation_system .
```

Run the container:

Please make sure the port 8510 is not occupied.

```bash
# Basic run
docker run -p 8510:8510 clothes_recommendation_system

# With volume mounts for data and configuration
docker run -p 8510:8510 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/user_info.py:/app/user_info.py \
  clothes_recommendation_system
```

## Logging

All logging is done using the `logger` object. To modify the logging configuration, edit the `src/logging_config.py` file.


## Example Query
"I am looking for a long white shirt."




## How the Search Engine Operates

1. Information Gathering and Client Inquiry
   - Ensuring the client has a clear understanding of their requirements
   - Posing clarifying questions if necessary

2. Keyword Selection for Search
   - Choosing keywords based on the client's information and the assistant's analysis

3. Shallow Search with Embedding Similarity
   - Rapidly identifying potential matches using vector embeddings
   - Applied across the entire dataset

4. Deep Search with LLM Verification
   - Validating and refining results using a Large Language Model
   - Conducted on the results of the shallow search

5. Leveraging Chat History for Accuracy
   - Utilizing past interactions to improve recommendation precision

6. Reasoning with Assistant and Client Data
   - Integrating insights from both the assistant's analysis and client data


## Feedback Mechanism

The feedback mechanism leverages textual data to iteratively refine the recommendation engine. By analyzing user interactions over time, the system adapts to individual preferences and styles, utilizing historical keyword data to enhance search accuracy and relevance.

Feedback with like or dislike will be converted to a text feedback. System generates thoughts based on the binary feedback by saving the details of products that user liked or disliked.


## Additional Features

1. Client Behavior Analysis
   - Identifies purchase intent through interaction patterns
   - Detects anomalous or suspicious behavior to ensure system integrity


