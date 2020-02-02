"""
    author: W J-H (jiangh_wu@163.com)
    time: Jan 9, 2020
    -----------------------------------
    数据读取接口
"""
import h5py
import json
import numpy as np


class DataLoader:
    def __init__(self, path_h5_img_train, path_h5_img_test, path_h5_ques, path_input_json, feat_type='VGG'):
        # h5文件惰性加载
        h5_img_train = h5py.File(path_h5_img_train, 'r')
        self.feat_img_train = h5_img_train.get('images_train')

        h5_img_test = h5py.File(path_h5_img_test, 'r')
        self.feat_img_test = h5_img_test.get('images_test')

        print("DataLoader loading h5 question file:", path_h5_ques)
        h5_ques = h5py.File(path_h5_ques, 'r')
        self.ques_train = h5_ques.get('ques_train')  # (N, 26)
        self.ques_len_train = h5_ques.get('ques_len_train')
        self.img_pos_train = h5_ques.get('img_pos_train')
        self.ques_id_train = h5_ques.get('ques_id_train')
        self.ans_train = h5_ques.get('answers')
        self.split_train = h5_ques.get("split_train")

        self.ques_test = h5_ques.get('ques_test')
        self.ques_len_test = h5_ques.get('ques_len_test')
        self.img_pos_test = h5_ques.get('img_pos_test')
        self.ques_id_test = h5_ques.get('ques_id_test')
        self.ans_test = h5_ques.get('ans_test')
        self.split_test = h5_ques.get('split_test')

        print("DataLoader loading js question file:", path_input_json)
        with open(path_input_json, 'r', encoding='utf-8') as f:
            js = json.load(f)
            self.id2word = js['ix_to_word']
            self.id2ans = js['ix_to_ans']

        self.seq_len = self.ques_train.shape[1]
        self.vocab_size = len(self.id2word)  # 预处理的id是从1开始的

        self.split_ix = {}  # 记录某个分割中的所有标号
        self.iterators = {}  # 记录某个分割迭代的位置

        for i, ix in enumerate(self.split_train):
            if ix not in self.split_ix:
                self.split_ix[ix] = []
                self.iterators[ix] = 0
            self.split_ix[ix].append(i)

        for i, ix in enumerate(self.split_test):
            if ix not in self.split_ix:
                self.split_ix[ix] = []
                self.iterators[ix] = 0
            self.split_ix[ix].append(i)

        for k, v in self.split_ix.items():
            print(f'assigned {len(v)} images to split {k}')

        self.img_feat_dim = 512 if feat_type == 'VGG' else 2048

    def reset_iter(self, split):
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_seq_len(self):
        return self.seq_len

    def get_ans_num(self):
        return len(self.id2ans)

    def get_data_num(self, split):
        return len(self.split_ix[split])

    def get_batch(self, split, batch_size=128):
        split_list = self.split_ix[split]

        max_index = len(split_list) - 1

        # 当前batch的图片索引
        img_id = np.zeros(batch_size)

        img_batch = np.zeros((batch_size, 14, 14, self.img_feat_dim), dtype='float32')
        ques_batch = np.zeros((batch_size, *self.ques_train[0].shape), dtype='uint16')
        ques_id_batch = np.zeros(batch_size, dtype='uint32')
        ques_len_batch = np.zeros(batch_size, dtype='uint32')
        ans_batch = np.zeros(batch_size, dtype='uint32')

        # 选取当前batch中的每个样本
        for i in range(batch_size):
            # 将迭代器往后走一步,并记录
            iter_now = self.iterators[split]
            iter_next = 0 if iter_now + 1 > max_index else iter_now + 1
            self.iterators[split] = iter_next

            # 获得当前样本的索引值
            if split < 2:  # 训练数据：随机挑选
                id_sample = np.random.choice(split_list)
                img_id[i] = self.img_pos_train[id_sample]

                img_batch[i] = self.feat_img_train[img_id[i]]
                ques_batch[i] = self.ques_train[id_sample]
                ques_id_batch[i] = self.ques_id_train[id_sample]
                ques_len_batch[i] = self.ques_len_train[id_sample]
                ans_batch[i] = self.ans_train[id_sample]

            else:  # 测试数据：顺序选择
                id_sample = split_list[iter_now]
                img_id[i] = self.feat_img_test[id_sample]

                img_batch[i] = self.feat_img_test[img_id[i]]
                ques_batch[i] = self.ques_test[id_sample]
                ques_id_batch[i] = self.ques_id_test[id_sample]
                ques_len_batch[i] = self.ques_len_test[id_sample]
                ans_batch[i] = self.ans_test[id_sample]

        batch_data = dict()
        batch_data['images'] = img_batch.reshape((batch_size, -1, self.img_feat_dim))  # (N, 196, dim)
        batch_data['questions'] = ques_batch
        batch_data['ques_id'] = ques_id_batch
        batch_data['ques_len'] = ques_len_batch
        batch_data['answers'] = ans_batch

        return batch_data


if __name__ == '__main__':
    dl = DataLoader("../data/cocoqa_data_img_vgg_train.h5",
                    "../data/cocoqa_data_img_vgg_test.h5",
                    "../data/cocoqa_data_prepro.h5",
                    "../data/backup/cocoqa_data_prepro.json")

    print(dl.get_batch(0, 32))
