import json
import os

import numpy as np
import random
import scipy.sparse as sp
from math import floor
import torch
from transformers import BertTokenizer, BertModel, BertConfig


class GraphAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def node_dropout(sp_adj, drop_rate):
        """Input: a sparse adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    @staticmethod
    def edge_dropout(sp_adj, drop_rate):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj


class SequenceAugmentor(object):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, '../LModel/RecBert')
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.config = BertConfig.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)

    def augview(seqs, seq_len, crop_ratio, call_count, weight):
        if isinstance(seqs, torch.Tensor):
            seqs = seqs.cpu().numpy()
        batch_size = seqs.shape[0]
        augmented_seq = np.zeros_like(seqs)
        augmented_pos = np.zeros_like(seqs)
        aug_len = []

        for i in range(batch_size):
            current_seq_len = random.randint(1, seq_len)
            start = random.randint(0, current_seq_len - floor(current_seq_len * crop_ratio))
            crop_len = floor(current_seq_len * crop_ratio) + 1
            if call_count % weight == 0:
                augmentor = SequenceAugmentor()
                seq_as_str = " ".join(map(str, seqs[i]))
                tokens = augmentor.tokenizer.encode_plus(
                    seq_as_str,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=seq_len
                )
                outputs = augmentor.model(**tokens)
                hidden_states = outputs[0]
                importance_scores = torch.mean(hidden_states, dim=-1).squeeze()
                adjustment_factor = importance_scores.mean().item()
                call_count += 1
            else:
                adjustment_factor = 0
            end = min(start + crop_len, current_seq_len)
            augmented_seq[i, :end - start] = seqs[i, start:end]
            augmented_pos[i, :end - start] = np.arange(1, end - start + 1) + adjustment_factor
            aug_len.append(end - start)
        call_count += 1
        return augmented_seq, augmented_pos, aug_len


if __name__ == '__main__':
    model_dir = "../LModel/RecBert/config.json"
    with open(model_dir, 'r') as f:
        config_content = json.load(f)
    print(config_content)