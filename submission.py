import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sim4rec.utils import pandas_to_spark

class GCNRecommender(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.gcn1 = nn.Linear(embedding_dim, embedding_dim)
        self.gcn2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, edge_index):
        x = self.embedding.weight
        x = F.relu(self.gcn1(x))
        x = self.gcn2(x)
        return x

class MyRecommender:
    def __init__(self, embedding_dim=64, epochs=5, lr=0.01, dropout=0.3, seed = None):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.item_prices = {}
        self.model = None

    def fit(self, log, user_features=None, item_features=None):
        df = log.toPandas()
        if item_features is not None:
            prices = item_features.select("item_idx", "price").toPandas()
            df = df.merge(prices, on="item_idx", how="left")

        self.num_users = df['user_idx'].max() + 1
        self.num_items = df['item_idx'].max() + 1
        self.num_nodes = self.num_users + self.num_items

        user_nodes = torch.tensor(df['user_idx'].values, dtype=torch.long)
        item_nodes = torch.tensor(df['item_idx'].values, dtype=torch.long) + self.num_users
        edge_index = torch.stack([
            torch.cat([user_nodes, item_nodes]),
            torch.cat([item_nodes, user_nodes])
        ], dim=0)

        pos_edges = list(zip(user_nodes.tolist(), item_nodes.tolist()))
        edge_set = set(pos_edges)

        neg_samples = []
        rng = np.random.default_rng(42)
        while len(neg_samples) < len(pos_edges):
            u = rng.integers(0, self.num_users)
            i = rng.integers(self.num_users, self.num_nodes)
            if (u, i) not in edge_set:
                neg_samples.append((u, i))

        pos_pairs = torch.tensor(pos_edges, dtype=torch.long)
        neg_pairs = torch.tensor(neg_samples, dtype=torch.long)
        labels = torch.cat([torch.ones(len(pos_pairs)), torch.zeros(len(neg_pairs))])
        train_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)

        self.model = GCNRecommender(self.num_nodes, self.embedding_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            self.model.train()
            embeddings = self.model(edge_index)
            user_emb = embeddings[train_pairs[:, 0]]
            item_emb = embeddings[train_pairs[:, 1]]
            scores = (user_emb * item_emb).sum(dim=1)
            loss = loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.embeddings = self.model(edge_index).detach()

        if item_features is not None:
            for _, row in prices.iterrows():
                self.item_prices[row['item_idx']] = row['price']

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        users = users.toPandas()
        items = items.toPandas()

        recs = []
        item_prices = torch.tensor([self.item_prices.get(i, 1.0) for i in items['item_idx']], dtype=torch.float)
        item_indices = torch.tensor(items['item_idx'].values, dtype=torch.long) + self.num_users
        item_embs = self.embeddings[item_indices]

        for uid in users['user_idx']:
            user_emb = self.embeddings[uid]
            scores = torch.sigmoid((user_emb * item_embs).sum(dim=1)) * item_prices
            top_k = torch.topk(scores, k)
            top_items = items.iloc[top_k.indices.numpy()]
            for i, row in top_items.iterrows():
                recs.append({
                    'user_idx': uid,
                    'item_idx': row['item_idx'],
                    'relevance': float(scores[top_k.indices[top_items.index.get_loc(i)]])
                })

        return pandas_to_spark(pd.DataFrame(recs))
