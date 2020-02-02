"""
    author: W J-H (jiangh_wu@163.com)
    time: Feb 2, 2020
    -----------------------------------
    辅助函数
"""

import torch.nn.functional as F


def masksoftmax(data, mask):
    data.masked_fill_(mask, -9999999)
    return F.softmax(data, dim=1)
