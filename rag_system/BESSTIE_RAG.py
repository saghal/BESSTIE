import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

class BESSTIE_RAG:
    def __init__(self, train_df, text_col="text", label_col="label"):
        self.__df = train_df.reset_index(drop=True)
        self.__text_col = text_col
        self.__label_col = label_col
        self.__encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.__embeddings = self.__encoder.encode(self.__df[text_col].tolist(), convert_to_numpy=True)
        faiss.normalize_L2(self.__embeddings)
        d = self.__embeddings.shape[1]

        # Creation of the index over the entire data
        self.__global_index = faiss.IndexFlatIP(d)
        self.__global_index.add(self.__embeddings)

        # Creation of two index for both negative and positive labeled data
        self.__neg_index = faiss.IndexFlatIP(d)
        self.__pos_index = faiss.IndexFlatIP(d)

        # Gather the index of the negative and positive entries
            # NEGATIVE
        self.__idx_0 = self.__df[self.__df[self.__label_col] == 0].index.to_numpy()
            # POSITIVE
        self.__idx_1 = self.__df[self.__df[self.__label_col] == 1].index.to_numpy()
        
        emb_0 = self.__embeddings[self.__idx_0]
        emb_1 = self.__embeddings[self.__idx_1]

        self.__neg_index.add(emb_0)
        self.__pos_index.add(emb_1)

    def get_query_embedding(self, query):
        emb = self.__encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    def __create_prompt(self, examples_df):
        prompt = "Here is examples of sentences and their labels : \n"
        i = 1
        for _, row in examples_df.iterrows():
            prompt += f"## Example {i}\n"
            prompt += f"Text: {row[self.__text_col]}\nLabel: {row[self.__label_col]}\n\n"
            i += 1
        return prompt

    def retrieve_closest(self, query, K=4):
        # Recover k examples globally based on the query (the label isn't important)
        query_emb = self.get_query_embedding(query)
        _, indices = self.__global_index.search(query_emb, K)
        examples = self.__df.iloc[indices[0]]

        # Return the prompt with the examples
        return self.__create_prompt(examples)

    def retrieve_contrastive(self, query, K=4):
        # We are going to create a prompt based on K/2 negative labeled entries, and K/2 positive labeled entries
        if K % 2 != 0:
            raise ValueError("K needs to be even.")
        
        K_2 = K // 2
        query_emb = self.get_query_embedding(query)
        
        # Gather the negative and positive entries
        _, indices_neg = self.__neg_index.search(query_emb, K_2)
        _, indices_pos = self.__pos_index.search(query_emb, K_2)

        # Combined the indices (the true one, from self.__embeddings) and shuffle the randomly for the prompt creation
        combined_indices = np.concatenate([self.__idx_0[indices_neg[0]], self.__idx_1[indices_pos[0]]])
        np.random.shuffle(combined_indices)
        examples = self.__df.iloc[combined_indices]

        # Return the prompt with the examples
        return self.__create_prompt(examples)

if (__name__ == "__main__"):
    # Usage examples
    train_df = pd.read_csv("train.csv")

    # Sentiment only
    train_df_sentiment = train_df[train_df["task"] == "Sentiment"]

    RAG_system = BESSTIE_RAG(train_df_sentiment)
    
    query = "I really enjoyed the movie"

    # Closest K sentences
    print(RAG_system.retrieve_closest(query, K=4))

    # Closest K/2 negative and K/2 positive sentences
    print(RAG_system.retrieve_contrastive(query, K=4))