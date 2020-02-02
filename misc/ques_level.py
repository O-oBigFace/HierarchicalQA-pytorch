import torch
import torch.nn as nn
from misc.alternating_attention import AlternatingAttention


class QuesLevel(nn.Module):
    def __init__(self, rnn_size, num_layers, dropout, hidden_size, seq_length, atten_type):
        super(QuesLevel, self).__init__()
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.atten_type = atten_type

        self.bilstm = nn.LSTM(input_size=self.hidden_size,
                              hidden_size=self.hidden_size // 2,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)

        if self.atten_type == 'Alternating':
            self.atten = AlternatingAttention(self.hidden_size,
                                              self.hidden_size,
                                              self.hidden_size,
                                              self.seq_length,
                                              196)

    def forward(self, ques_feat, img_feat, mask):
        self.bilstm.flatten_parameters()
        ques_feat, _ = self.bilstm(ques_feat)

        ques_att_q, img_att_q = self.atten(ques_feat, img_feat, mask)

        return ques_att_q, img_att_q


if __name__ == '__main__':
    qlevel = QuesLevel(512, 2, 0.5, 512, 26, 'Alternating')
    word_feat = torch.randn((10, 26, 512))
    img = torch.randn((10, 196, 512))
    mask = torch.zeros((10, 26), dtype=torch.bool)
    x = qlevel(word_feat, img, mask)
    print(x[0].size())

