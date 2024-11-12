import json
from collections import defaultdict
import math
import os

class TFIDF():
    def __init__(self):
        """
        Initialize TFIDF object
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

        # Handle empty tf-idf list and document scores files
        if os.stat('dataset/tf_idf_list.json').st_size == 0 or os.stat('dataset/ds.json').st_size == 0:
            self.construct_inverted_idx()
        
        # Load tf-idf list
        tf_idf_list_file = open('dataset/tf_idf_list.json')
        self.tf_idf_list = json.load(tf_idf_list_file)
        
        # Load document scores
        ds_file = open('dataset/ds.json')
        self.ds = json.load(ds_file)
    
    def construct_inverted_idx(self):
        """
        Construct inverted index
        """
        # Define data structure
        stats = {
            'words': {}, # key: a word, value: a set of docs containing that word
            'docs': {} # key: a word, value: frequency of that word in each doc
        }

        # Construct inverted index
        for i, doc in enumerate(self.docs):
            if i not in stats['docs']:
                stats['docs'][i] = defaultdict(int)
            
            for word in doc.split(' '):
                if word not in stats['words']:
                    stats['words'][word] = {i}
                else:
                    stats['words'][word].add(i)
                
                stats['docs'][i][word] += 1
        
        # Calculate idf
        idf = defaultdict(float) # inverse document frequency
        N = len(self.docs)
        words = stats['words'].keys()
        for word in words:
            df = len(stats['words'][word]) # document frequency
            idf[word] = math.log(N / df)
        
        tf_idf_list = defaultdict(lambda: defaultdict(float))
        ds = defaultdict(float)
        for doc in stats['docs']:
            d = 0
            for word in words:
                # Pre-calculating tf
                tf = self.__get_tf(stats['docs'][doc][word]) # term frequency
                # Calculate tf-idf
                tf_idf = tf * idf[word]
                d += tf_idf  ** 2
                # Store tf-idf value in tf_idf_list
                tf_idf_list[word][doc] = tf_idf
            
            d_ = d ** (1/2)
            # Store document score
            ds[doc] = self.__rounding(d_)
        
        # Save tf-idf list
        with open('tf_idf_list.json', 'w') as f:
            json.dump(tf_idf_list, f)
        # Save document scores
        with open('ds.json', 'w') as f:
            json.dump(ds, f)

    def search(self, q: str, k: int) -> list[tuple[float, int]]:
        """
        Search for documents containing query q
        :param q: str: query
        :param k: int: number of documents to be returned
        :return: list[tuple[float, int]]: list of top k documents with their scores
        """
        results = []

        # Loop through all documents
        for i in range(len(self.docs)):
            score = 0
            # Loop through all words in the query
            for word in q.split(' '):
                word = word.lower()
                score += self.tf_idf_list[word][i] / self.ds[i]
            # Update document score
            results.append((score, i))
        
        # Sort results by score in descending order
        results = sorted(results, key=lambda x: -x[0])

        # Return top k documents
        return results[:k]
    
    def __rounding(self, num: float) -> float:
        """
        Round a number
        :param num: float: number to be rounded
        :return: float: rounded number
        """
        return math.floor(num * 1000) / 1000
    
    def __get_tf(self, num: int) -> float:
        """
        Calculate term frequency
        :param num: int: frequency of a word in a document
        :return: float: term frequency
        """
        return self.__rounding(math.log10(num + 1))