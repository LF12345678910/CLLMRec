import torch

class GRU4Rec_withNeg_Dist(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GRU4Rec_withNeg_Dist, self).__init__()

        self.source_item_emb = torch.nn.Embedding(item_num + 1, args.hidden_units,
                                                  padding_idx=0)  # Embedding(3417, 50, padding_idx=0)
        self.target_item_emb = torch.nn.Embedding(item_num + 1, args.hidden_units,
                                                  padding_idx=0)  # Embedding(3417, 50, padding_idx=0)

        self.gru_source = torch.nn.GRU(args.hidden_units, args.hidden_units, batch_first=True)
        self.gru_target = torch.nn.GRU(args.hidden_units, args.hidden_units, batch_first=True)

        self.h0_source = torch.nn.Parameter(torch.zeros((1, 1, args.hidden_units), requires_grad=True))
        self.h0_target = torch.nn.Parameter(torch.zeros((1, 1, args.hidden_units), requires_grad=True))

        self.dev = args.device  # 'cuda'

        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()

        self.temperature = args.temperature
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_list, log_seqs_all, soft_diff):  # for training
        # user_ids:(128,)
        # log_seqs:(128, 200)
        # pos_seqs:(128, 200)
        # neg_seqs:(128, 200)
        #         ipdb.set_trace()
        neg_embs = []
        neg_logits = []
        source_log_embedding = self.source_item_emb(log_seqs)
        source_log_feats, _ = self.gru_source(source_log_embedding,
                                              self.h0_source.tile(1, source_log_embedding.shape[0], 1))  # 2，121，100
        source_log_all_embedding = self.source_item_emb(log_seqs_all)
        source_log_all_feats, _ = self.gru_source(source_log_all_embedding,
                                                  self.h0_source.tile(1, source_log_all_embedding.shape[0],
                                                                      1))  # 2，121，100
        pos_embs = self.source_item_emb(pos_seqs)  # torch.Size([128, 200, 50])
        soft_embs = self.source_item_emb(soft_diff)  # torch.Size([128, 200, 50])
        for i in range(0, len(neg_list)):
            neg_embs.append(self.source_item_emb(neg_list[i]))  # torch.Size([128, 200, 50])

        # get the l2 norm for the target domain recommendation
        source_log_feats_l2norm = torch.nn.functional.normalize(source_log_feats, p=2, dim=-1)
        pos_embs_l2norm = torch.nn.functional.normalize(pos_embs, p=2, dim=-1)
        pos_logits = (source_log_feats_l2norm * pos_embs_l2norm).sum(dim=-1)  # torch.Size([128, 200])
        pos_logits = pos_logits * self.temperature

        for i in range(0, len(neg_list)):
            neg_embs_l2norm_i = torch.nn.functional.normalize(neg_embs[i], p=2, dim=-1)
            neg_logits_i = (source_log_feats_l2norm * neg_embs_l2norm_i).sum(dim=-1)  # torch.Size([128, 200])
            neg_logits_i = neg_logits_i * self.temperature
            neg_logits.append(neg_logits_i)

        # 软标签对比
        source_log_all_feats_l2norm = torch.nn.functional.normalize(source_log_all_feats, p=2, dim=-1)
        soft_embs_l2norm = torch.nn.functional.normalize(soft_embs, p=2, dim=-1)
        soft_logits = (source_log_all_feats_l2norm[:, -1, :].unsqueeze(1).expand(-1, soft_embs_l2norm.shape[1],
                                                                                 -1) * soft_embs_l2norm).sum(
            dim=-1)  # torch.Size([128, 200])
        soft_logits = soft_logits * self.temperature

        return pos_logits, neg_logits, soft_logits  # pos_pred, neg_pred