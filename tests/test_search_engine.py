import pytest
import os
import sys
from unittest.mock import Mock, patch
import pandas as pd

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(root_dir, "src"))

from search_engine import SearchEngine

@pytest.fixture
def mock_openai():
    with patch('openai.Embedding.create') as mock_embed:
        # Mock embedding response
        mock_embed.return_value.data = [Mock(embedding=[0.1] * 1536)]
        yield mock_embed

@pytest.fixture
def mock_pinecone():
    with patch('pinecone.Pinecone') as mock_pc:
        # Mock Pinecone instance
        mock_pc.return_value.list_indexes.return_value.indexes = [{'name': 'test-index'}]
        mock_pc.return_value.Index.return_value.describe_index_stats.return_value = {"total_vector_count": 0}
        yield mock_pc

@pytest.fixture
def sample_data():
    # Create a small sample DataFrame
    return pd.DataFrame({
        'article_id': [1, 2],
        'prod_name': ['Test Product 1', 'Test Product 2'],
        'product_type_name': ['Type 1', 'Type 2'],
        'product_group_name': ['Group 1', 'Group 2'],
        'graphical_appearance_name': ['Solid', 'Pattern'],
        'colour_group_name': ['Red', 'Blue'],
        'perceived_colour_value_name': ['Light', 'Dark'],
        'perceived_colour_master_name': ['Red', 'Blue'],
        'department_name': ['Dept 1', 'Dept 2'],
        'index_name': ['Index 1', 'Index 2'],
        'index_group_name': ['Group 1', 'Group 2'],
        'section_name': ['Section 1', 'Section 2'],
        'garment_group_name': ['Garment 1', 'Garment 2'],
        'detail_desc': ['Description 1', 'Description 2']
    })

class TestSearchEngine:
    
    def test_initialization(self, mock_openai, mock_pinecone, sample_data, tmp_path):
        """Test SearchEngine initialization"""
        # Create a temporary CSV file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sample_data.to_csv(data_dir / "articles.csv", index=False)
        
        # Initialize SearchEngine
        engine = SearchEngine(init_vector_database=True, data_dir=str(data_dir))
        
        assert engine.embedding_model_name == "text-embedding-ada-002"
        assert engine.embedding_dim == 1536
        assert isinstance(engine.table, pd.DataFrame)
        assert len(engine.table) == len(sample_data)

    def test_row_to_text(self, mock_openai, mock_pinecone, sample_data, tmp_path):
        """Test row_to_text conversion"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sample_data.to_csv(data_dir / "articles.csv", index=False)
        
        engine = SearchEngine(init_vector_database=True, data_dir=str(data_dir))
        
        # Test text conversion for first row
        text = engine.row_to_text(sample_data.iloc[0])
        assert isinstance(text, str)
        assert "Test Product 1" in text
        assert "Type 1" in text

    @patch('openai.ChatCompletion.create')
    def test_verify_search_result_relevance(self, mock_chat, mock_openai, mock_pinecone, sample_data, tmp_path):
        """Test search result verification"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sample_data.to_csv(data_dir / "articles.csv", index=False)
        
        # Mock ChatCompletion response
        mock_chat.return_value.choices = [
            Mock(message={'content': 'true\nRelevant because it matches the query'})
        ]
        
        engine = SearchEngine(init_vector_database=True, data_dir=str(data_dir))
        
        # Test verification
        result = {'metadata': {'prod_name': 'Test Product'}}
        is_relevant, reason = engine.verify_search_result_relevance("red dress", result)
        
        assert isinstance(is_relevant, bool)
        assert isinstance(reason, str)

    def test_embedding_search(self, mock_openai, mock_pinecone, sample_data, tmp_path):
        """Test embedding search functionality"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sample_data.to_csv(data_dir / "articles.csv", index=False)
        
        # Mock Pinecone query response
        mock_matches = [
            {
                'id': '1',
                'score': 0.9,
                'metadata': {'prod_name': 'Test Product 1'}
            }
        ]
        mock_pinecone.return_value.Index.return_value.query.return_value.matches = mock_matches
        
        engine = SearchEngine(init_vector_database=True, data_dir=str(data_dir))
        
        # Test search
        with patch.object(engine, 'verify_search_result_relevance', return_value=(True, "Relevant")):
            results = engine.embedding_search("test query")
            assert isinstance(results, list)
            assert len(results) > 0

    def test_get_image_path(self, mock_openai, mock_pinecone, sample_data, tmp_path):
        """Test image path generation"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sample_data.to_csv(data_dir / "articles.csv", index=False)
        
        engine = SearchEngine(init_vector_database=True, data_dir=str(data_dir))
        
        # Test path generation
        path = engine.get_image_path(12345)
        assert isinstance(path, str)
        assert "012" in path
        assert "012345.jpg" in path
