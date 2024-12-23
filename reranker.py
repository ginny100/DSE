import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset

import os

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = AutoModel.from_pretrained('vinai/phobert-base-v2')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')

def get_embedding(item: str) -> torch.Tensor:
    """
    Get embedding of an item
    :param item: str: item to get embedding
    :return: torch.Tensor: mean pooled embedding of the item
    """
    # Tokenize the item
    tokens = tokenizer(item, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    # Get the embeddings
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    # Mask the embeddings
    emb_size = embeddings.size()
    mask = tokens['attention_mask'].unsqueeze(-1).expand(emb_size).float()
    masked_embeddings = mask * embeddings 
    # Get the mean-pooled embeddings
    summed_embeddings = torch.sum(masked_embeddings, 1)
    counted = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed_embeddings / counted
    print('mean_pooled shape:', mean_pooled.shape)
    return mean_pooled.detach().numpy()

class ReRanker():
    def __init__(self):
        """
        Initialize ReRanker object
        """
        self.load_data()
    
    def load_data(self):
        """
        Load data from json files
        """
        # Load documents
        docs_file = open('dataset/docs.json')
        data = json.load(docs_file)
        self.docs = data['docs']

    def rank(self, query: str, docs: list[tuple[float, int]]) -> list[dict]:
        """
        Re-rank the documents based on the query
        :param query: str: query
        :param docs: list[tuple[float, int]]: list of documents with their scores
        :return: list[dict]: list of re-ranked documents with their scores
        """
        # Load docs from TF-IDF and convert to Hugging Dataset
        dataset = Dataset.from_list([
            { 'id': int(doc[1]), 'text': self.docs[int(doc[1])] } for doc in docs
        ])
        
        # Map dataset to include embeddings
        dataset_embedding = dataset.map(
            lambda example: {'embeddings': get_embedding(example['text'])[0]}
        )
        # Add FAISS index to 'embeddings' column
        dataset_embedding.add_faiss_index(column='embeddings')
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Search for vector similarity with query
        scores, retrieved_examples = dataset_embedding.get_nearest_examples('embeddings', query_embedding, k=10)
        print('scores:', scores)
        print('retrieved_examples:', retrieved_examples)
        
        # Sort retrieved examples by scores in descending order
        examples_df = pd.DataFrame.from_dict(retrieved_examples)
        examples_df['scores'] = scores
        examples_df.sort_values('scores', ascending=False, inplace=True)

        # Get the results
        results = []
        for _, row in examples_df.iterrows():
            print(row['scores'], row['text'])
            results.append({
                'id': row['id'],
                'text': row['text'],
                'score': row['scores']
            })
        return results