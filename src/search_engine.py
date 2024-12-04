import os, sys

########################################################
# add the data directory to the system path
########################################################

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "data"))

########################################################
# imports
########################################################

import pandas as pd
import pdb
from tqdm import tqdm
import time

########################################################
# OpenAI
########################################################

import openai
from user_info import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

########################################################
# Pinecone
########################################################

from pinecone import Pinecone
from pinecone import ServerlessSpec

from user_info import PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_CLOUD

PC = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

class SearchEngine:

    def __init__(self, init_vector_database: bool, data_dir: str = os.path.join(root_dir, "data")):
        self.embedding_model_name = "text-embedding-ada-002"
        self.index_name = PINECONE_INDEX_NAME

        self.data_dir = data_dir
        self.query_history = []
        n_selection = 200    

        response = openai.Embedding.create(
            model=self.embedding_model_name,
            input="",
        )

        self.embedding_dim = len(response.data[0].embedding)

        ########################################################  
        # generating search text and embeddings
        ########################################################

        self.table = pd.read_csv(os.path.join(self.data_dir, "articles.csv"))

        if n_selection is not None:
            self.table = self.table[:n_selection]

        tqdm.pandas(desc="Creating search text")
        self.table["search_text"] = self.table.progress_apply(self.row_to_text, axis=1)

        embeddings = self.embed_text_list(self.table["search_text"].tolist())
        self.table["embedding"] = embeddings

        ########################################################
        # vector database
        ########################################################

        existing_indexes_info = PC.list_indexes()
        existing_index_names = [index['name'] for index in existing_indexes_info.indexes]

        if self.index_name not in existing_index_names:
            PC.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_ENVIRONMENT,
                ),
            )

        self.vector_database = PC.Index(self.index_name)   

        if init_vector_database:
            # delete index if it exists
            if self.index_name in existing_index_names:
                if self.vector_database.describe_index_stats()["total_vector_count"] > 0:
                    self.vector_database.delete(delete_all=True)

            meta_columns = list(set(self.table.columns.tolist()) - set(["embedding"]))  

            vectors_to_upsert = [
                {
                    "id": str(getattr(row, "article_id")),
                    "values": embd.embedding,
                    "metadata": {col: getattr(row, col) for col in meta_columns if col != "article_id"},
                }
                for row, embd in zip(self.table.itertuples(), embeddings)
            ]

            batch_size = 100
        
            for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upserting vectors"):
                self.vector_database.upsert(
                    vectors=vectors_to_upsert[i:i+batch_size]
                )

            print(f"Database status: {self.vector_database.describe_index_stats()}")

        ########################################################  
        # embedding dimension
        ########################################################

        self.embedding_dim = len(self.table["embedding"].iloc[0].embedding)
    
        ########################################################
        # Check some samples from the database
        ########################################################

        if False:
            result = self.vector_database.query(
                vector=[0] * self.embedding_dim,    
                top_k=10,                           
                include_metadata=True               
            )

            for idx, match in enumerate(result['matches']):
                print(f"Sample {idx + 1}:")
                print(f"ID: {match['id']}")
                print(f"Score: {match['score']}")
                print(f"Metadata: {match.get('metadata', {})}")
                print("-" * 50)

    def row_to_text(self, row: pd.Series):
        selected_columns = [
            "prod_name",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
            "detail_desc"
        ]

        text = ""
        for col in selected_columns:
            text += f"{col}: {row[col]}\n"

        return text

    ########################################################
    # Embeddings
    ########################################################

    def database_status(self):
        print(f"Database status: {self.vector_database.describe_index_stats()}")

    ########################################################
    # Embeddings
    ########################################################

    def embed_text(self, text: str):
        response = openai.Embedding.create(
            model=self.embedding_model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_text_list(self, text_list: list, batch_size: int = 1000):
        embeddings = []

        for i in tqdm(range(0, len(text_list), batch_size), desc="Generating embeddings"):
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model_name,
                    input=text_list[i:i+batch_size],
                )
                embeddings.extend(response.data)
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                raise e
                    
        return embeddings

    def get_image_path(self, article_id: int):
        article_id = "0" + str(article_id)
        code = str(article_id)[:3]
        path = os.path.join(self.data_dir, "images", code, f"{article_id}.jpg")
        return path

    ########################################################
    # Search
    ########################################################

    def embedding_search(self, query: str, k_top: int = 10):
        query_embedding = self.embed_text(query)

        result = self.vector_database.query(
            vector=query_embedding,
            top_k=k_top,
            include_metadata=True
        ) 

        return result

    ########################################################
    # Re-Ranking
    ########################################################

    def semantic_score(self, query: str, related_article_ids: list):
        pass

    ########################################################
    # Think
    ########################################################

    ########################################################
    # query_history
    ########################################################