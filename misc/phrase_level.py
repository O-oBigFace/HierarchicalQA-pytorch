import torch
import torch.nn as nn
from misc.alternating_attention import AlternatingAttention


class PhraseLevel(nn.Module):
    def __init__(self, hidden_size, seq_length, atten_type):
        super(PhraseLevel, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.atten_type = atten_type

        self.conv_unigram = nn.Conv1d(in_channels=self.hidden_size,
                                      out_channels=self.hidden_size,
                                      kernel_size=1,
                                      stride=1)
        self.conv_bigram = nn.Conv1d(in_channels=self.hidden_size,
                                     out_channels=self.hidden_size,
                                     kernel_size=2,
                                     padding=1)
        self.conv_trigram = nn.Conv1d(in_channels=self.hidden_size,
                                      out_channels=self.hidden_size,
                                      kernel_size=3,
                                      padding=1)

        if self.atten_type == 'Alternating':
            self.atten = AlternatingAttention(self.hidden_size,
                                              self.hidden_size,
                                              self.hidden_size,
                                              self.seq_length,
                                              196)

    def forward(self, word_feat, img_feat, mask):
        # phrase feat
        word_feat_per = word_feat.permute(0, 2, 1)
        unigram = self.conv_unigram(word_feat_per)

        bigram = self.conv_bigram(word_feat_per)
        bigram = bigram.narrow(2, 0, 26)

        trigram = self.conv_trigram(word_feat_per)

        unigram = unigram.permute(0, 2, 1).view(-1, self.seq_length, self.hidden_size, 1)
        bigram = bigram.permute(0, 2, 1).view(-1, self.seq_length, self.hidden_size, 1)
        trigram = trigram.permute(0, 2, 1).view(-1, self.seq_length, self.hidden_size, 1)

        context_feat = torch.cat((unigram, bigram, trigram), dim=-1)
        context_feat = torch.max(context_feat, -1)[0]

        ques_att_p, img_att_p = self.atten(context_feat, img_feat, mask)

        return ques_att_p, img_att_p

