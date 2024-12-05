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

Run the test suite using pytest:
```bash
pytest tests/test_search_engine.py
pytest tests/test_app.py
```

## Docker Support

Build the Docker image:
```bash
docker build -t clothes_recommendation_system .
```

Run the container:
```bash
# Basic run
docker run -p 8510:8510 clothes_recommendation_system

# With volume mounts for data and configuration
docker run -p 8510:8510 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/user_info.py:/app/user_info.py \
  clothes_recommendation_system
```

## Example Query
"I am looking for a long white shirt."

