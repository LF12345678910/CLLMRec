import torch
import torch.nn as nn

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class SASRec_Model(nn.Module):
    def __init__(self, item_num, emb_size, max_len, n_blocks, n_heads, drop_rate):
        super(SASRec_Model, self).__init__()
        self.item_num = item_num
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        # self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num+1, self.emb_size)))
        self.item_emb = nn.Parameter(initializer(torch.empty(self.item_num + 1, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len +1, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer =  torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seq, pos):
        seq_emb = self.item_emb[seq]
        seq_emb *= self.emb_size ** 0.5

        # print("======================")
        # # print(self.item_emb.shape)
        # # print(self.pos_emb.shape)
        # print(seq.shape)
        # print(pos.shape)
        # print(pos)
        # print(f"seq min: {seq.min()}, seq max: {seq.max()}")
        # print(f"pos min: {pos.min()}, pos max: {pos.max()}")
        pos_emb = self.pos_emb[pos]
        seq_emb += pos_emb

        # seq_cpu = seq.cpu()  # 将seq移到CPU
        # print(f"seq min: {seq_cpu.min()}, seq max: {seq_cpu.max()}")

        # print(f"seq min: {seq.min()}, seq max: {seq.max()}")
        # seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = (seq == 0)  # 生成布尔数组
        timeline_mask = torch.tensor(timeline_mask, dtype=torch.bool).cuda()  # 将其转换为 CUDA 上的布尔张量
        # timeline_mask = torch.BoolTensor(seq == 0).cuda()

        seq_emb *= ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            #attention_input = seq_emb
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb *=  ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate,activation='relu'):
        super(PointWiseFeedForward, self).__init__()
        act = None
        if activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'gelu':
            act = torch.nn.GELU()
        self.pwff = torch.nn.Sequential(
            torch.nn.Linear(hidden_units, hidden_units),
            #torch.nn.Dropout(p=dropout_rate),
            act,
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.pwff(inputs)
        outputs += inputs
        return outputs

