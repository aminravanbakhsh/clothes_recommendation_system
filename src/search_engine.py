import os, sys

########################################################
# add the data directory to the system path
########################################################

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "data"))

########################################################
# logging
########################################################

from logging_config import setup_logger
logger = setup_logger(__name__)

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


########################################################
# SearchEngine
########################################################

class SearchEngine:

    def __init__(self, init_vector_database: bool, data_dir: str = os.path.join(root_dir, "data")):

        logger.info(f"Initializing SearchEngine with data_dir: {data_dir}")

        self.embedding_model_name   = "text-embedding-ada-002"
        self.embedding_dim          = 1536
        self.index_name             = PINECONE_INDEX_NAME

        self.data_dir               = data_dir
        self.query_history          = []
        n_selection                 = 10000

        ########################################################  
        # generating search text 
        ########################################################

        logger.info(f"Loading articles from {os.path.join(self.data_dir, 'articles.csv')}")

        self.table = pd.read_csv(os.path.join(self.data_dir, "articles.csv"))

        if (n_selection is not None) and (n_selection > len(self.table)):
            self.table = self.table[:n_selection]

        tqdm.pandas(desc="Creating search text")
        self.table["search_text"] = self.table.progress_apply(self.row_to_text, axis=1)


        ########################################################
        # vector database
        ########################################################

        existing_indexes_info = PC.list_indexes()
        existing_index_names = [index['name'] for index in existing_indexes_info.indexes]

        logger.info(f"Checking for existing index: {self.index_name}")

        if self.index_name not in existing_index_names:
            logger.info(f"Index not found. Creating new index")

            logger.info(f"Creating new Pinecone index: {self.index_name}")
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
            logger.info(f"Initializing vector database")

            
            # delete index if it exists
            if self.index_name in existing_index_names:
                if self.vector_database.describe_index_stats()["total_vector_count"] > 0:
                    self.vector_database.delete(delete_all=True)

            meta_columns = list(set(self.table.columns.tolist()))  


            if n_selection < len(self.table):
                embeddings = self.embed_text_list(self.table["search_text"].tolist()[:n_selection])
            else:
                embeddings = self.embed_text_list(self.table["search_text"].tolist())

            vectors_to_upsert = [
                {
                    "id": str(getattr(row, "article_id")),
                    "values": embd.embedding,
                    "metadata": {
                        col: str(getattr(row, col)) if pd.notna(getattr(row, col)) else "N/A" 
                        for col in meta_columns 
                        if col != "article_id"
                    },
                }
                for row, embd in zip(self.table.itertuples(), embeddings)
            ]

            batch_size = 100
        
            logger.info(f"Upserting {len(vectors_to_upsert)} vectors to database")
            for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upserting vectors"):
                self.vector_database.upsert(
                    vectors=vectors_to_upsert[i:i+batch_size]
                )

            print(f"Database status: {self.vector_database.describe_index_stats()}")

    
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

        logger.info(f"Generating embeddings for {len(text_list)} texts")

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
                    
        logger.info(f"Generated embeddings for {len(embeddings)} texts")

        return embeddings

    def get_image_path(self, article_id: int):
        article_id = "0" + str(article_id)
        code = str(article_id)[:3]
        path = os.path.join(self.data_dir, "images", code, f"{article_id}.jpg")
        return path

    ########################################################
    # Search
    ########################################################

    def extract_search_material(self, conversation: list) -> str:
        logger.info(f"Extracting search material from query: {conversation}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "assistant", "content": """You are a clothing search relevance checker, used for a retrieval system.
                Your task is to determine key words in the the conversation for a search query. Pay more attention to the last message from the user. 
                If first message are different from the last message, ignore them. But if they are relevant, include them. For example:
                change in color, style, type, etc.
                return a list of keywords, separated by commas.
                """},
                {"role": "user", "content": f"Query: {conversation}"},
            ]
        )

        keywords = response.choices[0].message.content.strip()
        logger.info(f"Extracted keywords: {keywords}")
            
        return keywords

    def embedding_search(self, query: str, k_top: int = 10):

        logger.info(f"Performing embedding search for query: {query}")

        query_embedding = self.embed_text(query)

        result = self.vector_database.query(
            vector=query_embedding,
            top_k=k_top,
            include_metadata=True
        ) 

        verified_results = []
        for match in result.matches:
            is_relevant, reason = self.verify_search_result_relevance(query, match)
            if is_relevant:
                verified_results.append(match)
            else:
                print(f"Result {match['id']} is not relevant. Reason: {reason}")

        logger.info(f"Found {len(verified_results)} relevant results")
        result.matches = verified_results

        # pdb.set_trace()

        return verified_results

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

    ########################################################
    # verify_search_result_relevance
    ########################################################

    def verify_search_result_relevance(self, query: str, result: dict):

        logger.info(f"Verifying relevance for result based on query. query: {query} result id: {result['id']}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "assistant", "content": f"""You are a clothing search relevance checker, used for a retrieval system.
                Your task is to determine if a search result matches the user's query. Pay more attention to the last message from the user.
                Respond with ONLY two lines:
                Line 1: Either 'true' or 'false'
                Line 2: A brief explanation of your decision
                Query: \"{query}\"\nResult: \"{result}\""""}
            ]
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
            logger.error(f"Invalid response from OpenAI: {result}")
            raise ValueError(f"Invalid response from OpenAI: {result}")
