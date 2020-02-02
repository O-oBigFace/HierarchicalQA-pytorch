"""
    author: W J-H (jiangh_wu@163.com)
    time: Jan 8, 2020
    -----------------------------------
    单词级别的文本特征
"""
import torch
import torch.nn as nn
from misc.alternating_attention import AlternatingAttention


class WordLevel(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length, atten_type, feature_type):
        super(WordLevel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.atten_type = atten_type
        self.img_feat_dim = 512 if feature_type == 'VGG' else 2048

        self.language_embedding = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.hidden_size),
                                                nn.Tanh(),
                                                nn.Dropout(0.5))  # (N, seq_len, hidden_size)

        self.img_linear = nn.Sequential(
            nn.Linear(self.img_feat_dim, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.5))

        if self.atten_type == 'Alternating':
            self.atten = AlternatingAttention(self.hidden_size,
                                              self.hidden_size,
                                              self.hidden_size,
                                              self.seq_length,
                                              196)

    def forward(self, seq, img):
        """

            ------------------------------------------
            Args:
                seq: (N, seq_len)
                img: (N, )
            Returns:
        """
        word_feat = self.language_embedding.forward(seq)  # (N, seq_len, dim)

        mask = torch.zeros_like(seq, dtype=torch.bool)
        mask[torch.eq(seq, 0)] = True

        img_feat = self.img_linear(img)

        ques_att_w, img_att_w = self.atten(word_feat, img_feat, mask)

        return ques_att_w, img_att_w, (word_feat, img_feat, mask)
