"""
    author: W J-H (jiangh_wu@163.com)
    time: Jan 8, 2020
    -----------------------------------
"""
import torch
import torch.nn as nn
from misc.helper import masksoftmax


class AlternatingAttention(nn.Module):
    def __init__(self, input_ques_dim, input_img_dim, embedding_dim, ques_seq_len, img_seq_len):
        super(AlternatingAttention, self).__init__()
        self.input_ques_dim = input_ques_dim
        self.input_img_dim = input_img_dim
        self.embedding_dim = embedding_dim
        self.ques_seq_len = ques_seq_len
        self.img_seq_len = img_seq_len

        self.dense_h1 = nn.Linear(self.input_ques_dim, self.embedding_dim)
        self.dense_a1 = nn.Linear(self.embedding_dim, 1)
        self.dense_h2_1 = nn.Linear(self.input_img_dim, self.embedding_dim)
        self.dense_h2_2 = nn.Linear(self.input_ques_dim, self.embedding_dim)
        self.dense_a2 = nn.Linear(self.embedding_dim, 1)
        self.dense_h3_1 = nn.Linear(self.input_ques_dim, self.embedding_dim)
        self.dense_h3_2 = nn.Linear(self.input_img_dim, self.embedding_dim)
        self.dense_a3 = nn.Linear(self.embedding_dim, 1)

    def forward(self, ques_feat, img_feat, mask):
        """
            AlternatingAttention
            主要实现以下公式:
                H = tanh(W_x * X + (W_g * g) * 1^T)
                alpha^x = softmax(W_{hx}^T * H)
                x_hat = \sum alpha^x_i * x_i
            ------------------------------------------
            Args:
                ques_feat: (N, seq_len, input_embedding_dim)
                img_feat:
                mask: 将padding的位置屏蔽
            Returns:
        """
        # iter 1
        # H1 = tanh(X * W_x)
        h1 = self.dense_h1(ques_feat)
        h1 = nn.Dropout(0.5)(nn.Tanh()(h1))  # (N, seq_len, embedding_dim)

        # alpha^x = softmax(H1 * W_{hx})
        a1 = self.dense_a1(h1)  # (N, seq_len, 1)
        a1 = masksoftmax()(a1.view(-1, self.ques_seq_len), mask)  # (N, seq_len)

        # s1 = \sum alpha^x_i * x_i
        s1 = a1.unsqueeze(-1) * ques_feat
        s1 = torch.sum(s1, dim=1)  # (N, input_ques_dim)

        # iter2
        # H2 = tanh(X * W_x + (W_g * s1) * 1^T)
        h2_1 = self.dense_h2_1(img_feat)  # (N, img_seq_len, embedding_dim)
        h2_2 = self.dense_h2_2(s1)
        h2_2 = h2_2.unsqueeze(1).expand(-1, self.img_seq_len, self.embedding_dim)  # the same
        h2 = nn.Dropout(0.5)(nn.Tanh()(h2_1 + h2_2))

        # alpha2
        a2 = self.dense_a2(h2)
        a2 = nn.Softmax(1)(a2.view(-1, self.img_seq_len))

        # v2
        v2 = a2.unsqueeze(-1) * img_feat
        v2 = torch.sum(v2, dim=1)

        # iter3
        h3_1 = self.dense_h3_1(ques_feat)
        h3_2 = self.dense_h3_2(v2)
        h3_2 = h3_2.unsqueeze(1).expand(-1, self.ques_seq_len, self.embedding_dim)
        h3 = nn.Dropout(0.5)(nn.Tanh()(h3_1 + h3_2))

        # alpha3
        a3 = self.dense_a3(h3)
        a3 = nn.Softmax(1)(a3.view(-1, self.ques_seq_len))

        s3 = a3.unsqueeze(-1) * ques_feat
        s3 = torch.sum(s3, dim=1)

        return s3, v2
