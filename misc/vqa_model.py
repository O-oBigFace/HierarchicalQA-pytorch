"""
    author: W J-H (jiangh_wu@163.com)
    time: Jan 14, 2020
    -----------------------------------
    模型整体
"""
import torch.nn as nn
from misc.word_level import WordLevel
from misc.phrase_level import PhraseLevel
from misc.ques_level import QuesLevel
from misc.recursive_encoder import RecursiveEncoder


class VQAModel(nn.Module):
    def __init__(self, opt: dict):
        super(VQAModel, self).__init__()

        self.word_model = WordLevel(
            opt['vocab_size'],  #
            opt['hidden_size'],
            opt['seq_length'],  #
            opt['co_atten_type'],
            opt['feature_type'])

        self.phrase_model = PhraseLevel(
            opt['hidden_size'],
            opt['seq_length'],
            opt['co_atten_type'])

        self.ques_model = QuesLevel(
            opt['rnn_size'],
            opt['rnn_layers'],
            opt['dropout'],
            opt['hidden_size'],
            opt['seq_length'],
            opt['co_atten_type'])

        self.recursive_encoder = RecursiveEncoder(
            opt['hidden_size'],
            opt['hidden_size'],
            opt['hidden_size'],
            opt['output_size'])

    def forward(self, questions, images):
        ques_att_w, img_att_w, (word_feat, img_feat, mask) = self.word_model(questions, images)
        ques_att_p, img_att_p = self.phrase_model(word_feat, img_feat, mask)
        ques_att_q, img_att_q = self.ques_model(word_feat, img_feat, mask)

        out = self.recursive_encoder(ques_att_w, img_att_w, ques_att_p, img_att_p, ques_att_q, img_att_q)

        return out
