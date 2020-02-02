"""
    author: W J-H (jiangh_wu@163.com)
    time: Jan 5, 2020
    -----------------------------------
    对数据进行进一步预处理：详情见函数train_data_process, test_data_process的说明
"""
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
import json
import h5py
import re


def get_top_ans(exps: list):
    """
        获取所有的答案
        ------------------------------------------
        Args:
            exps: list, 每个元素可以看成一个数据
                [
                  {
                    'ques_id': int id,
                    'img_path': path,
                    'question': question,
                    'types': question type,
                    'ans': answer
                  },
                  {...},
                  ...
                ]
        Returns:
            vocab: list, 每一项都是一个答案
    """
    count = dict()
    for exp in exps:
        ans = exp['ans']
        count[ans] = count.get(ans, 0) + 1

    rank = sorted([(n, a) for a, n in count.items()], reverse=True)
    print("The top answer and their counts:")
    print("\n".join(map(str, rank[:20])))

    return [r[1] for r in rank]


def encode_answer(exps, ans2id):
    """
        将exps中的答案id转化为张量编码
        暂时地，将不在ans2id中的答案设置为0
        ------------------------------------------
        Args:
        Returns:
    """
    res = np.zeros(len(exps), dtype="uint32")

    for i, exp in enumerate(exps):
        if exp["ans"] in ans2id:
            res[i] = ans2id[exp['ans']]

    return res


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence)
            if i != '' and i != ' ' and i != '\n']


def tokenize_questions(exps, params):
    """
        对问题进行tokenize
        ------------------------------------------
        Args:
            exps:
            params:
        Returns:
            对dict的操作，无需返回
    """
    N = len(exps)
    for i, exp in enumerate(exps):
        if params["token_method"] == "nltk":
            tokens = word_tokenize(exp["question"].lower())
        else:
            tokens = tokenize(exp["question"].lower())

        exp["processed_tokens"] = tokens

        if i % 10000 == 0 or i == N-1:
            print(f"Tokenization processing {i+1}/{N}, ({ (i+1) * 100 / N:.2f}% is done)")


def build_vocab_questions(exps: list, params: dict):
    """
        为问题tokens构建词典:
        1. 提取问题中的所有单词
        2. 过滤出现次数小于阈值的词
        3. 将问题中出现的频次过低的词赋值为UNK
        ------------------------------------------
        Args:
            exps: 同本文件其他
            params: 同
        Returns:
            vocab: 过滤后的词汇表
    """
    theshold = params["word_count_threshold"]

    # 1.
    counts = {}
    for exp in exps:
        for token in exp["processed_tokens"]:
            counts[token] = counts.get(token, 0) + 1

    rank = sorted([(c, w) for w, c in counts.items()], reverse=True)
    print("the top words:")
    print("\n".join(map(str, rank[:20])))

    # 2.
    total_count = sum(counts.values())
    bad_words = [w for (c, w) in rank if c <= theshold]
    vocab = [w for (c, w) in rank if c > theshold]
    bad_count = sum([counts[bw] for bw in bad_words])
    print(f"number of bad words: {len(bad_words)}/{len(counts)} = {len(bad_words) * 100/ len(counts):.2f}%")
    print(f"number of words in vocab would be {len(vocab)}")
    print(f"number of UNKs: {bad_count}/{total_count} = {bad_count * 100 / total_count}%")

    # 3.
    vocab.append("UNK")
    for exp in exps:
        exp["final_question"] = [w if w not in bad_words else "UNK" for w in exp["processed_tokens"]]

    return vocab


def encode_questions(exps, params, word2id):
    """
        将问题tokens转化为张量表示:
        1. 问题列表-->二维张量
        2. 问题长度-->一维张量
        3. 问题id --> 一维张量
        ------------------------------------------
        Args:
            exps: 同上
            params: 同上
            word2id: train_data_process中获得
        Returns:

    """
    max_len = params["max_length"]
    N = len(exps)

    question_length = np.array([min(len(exp["final_question"]), max_len) for exp in exps], dtype="uint32")
    question_id = np.array([exp["ques_id"] for exp in exps], dtype="uint32")
    question_arrays = np.zeros((N, max_len), dtype="uint32")

    for i, exp in enumerate(exps):
        for k, w in enumerate(exp["final_question"]):
            if k < max_len:
                question_arrays[i, k] = word2id[w]

    return question_arrays, question_length, question_id


def get_unique_img(exps):
    """
        获取数据中所有会用到的图片路径，并赋给图片一个id.
        ------------------------------------------
        Args:
            exps: 同上
        Returns:
            unq_imgs: list, 包含涉及到的所有图片的路径，没有重复项，并且顺序是固定的
            img_path: list, 对应每一个exp，将图片路径转化为编码
    """
    img2id = dict()
    count = 0
    unq_imgs = list()
    for exp in exps:
        if exp['img_path'] not in img2id:
            img2id[exp['img_path']] = count
            unq_imgs.append(exp['img_path'])

            count += 1

    img_id = [img2id[exp["img_path"]] for exp in exps]

    return unq_imgs, np.array(img_id, dtype="uint32")


def train_data_process(params, out_dict):
    """
        对训练数据进行预处理的整合函数:
        1. 处理答案：统计，答案->id，将exp中的答案转化为id, 对答案张量化
        2. 处理问题：tokenize, 统计词汇，过滤低频词，将问题转化为包含索引的张量
        3. 赋涉及图片以id
        4. 蜜汁split，到底干嘛的
        5. 将张量存入h5文件，将词典置入out_dict
        ------------------------------------------
        Args:

        Returns:
    """
    exps = json.load(open(params['input_train_json'], 'r', encoding='utf-8'))

    # 1.
    top_ans = get_top_ans(exps)  # 处理答案
    ans2id = {a: i for i, a in enumerate(top_ans)}
    id2ans = {i: a for a, i in ans2id.items()}
    ans_array = encode_answer(exps, ans2id)

    # 2.
    tokenize_questions(exps, params)
    vocab = build_vocab_questions(exps, params)
    word2id = {w: i+1 for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}

    # 3.
    question_array, question_length_array, question_id_array = encode_questions(exps, params, word2id)

    # 4.
    unique_img, img_path_array = get_unique_img(exps)

    # 5.
    # split = np.zeros(len(exps))

    # 6.
    h5file = h5py.File(params['output_train_h5'], 'w')

    h5file.create_dataset("ans", dtype='int64', data=ans_array)

    h5file.create_dataset("ques", dtype='int64', data=question_array)

    h5file.create_dataset("ques_id", dtype='int64', data=question_id_array)

    h5file.create_dataset("img_pos", dtype='int64', data=img_path_array)

    h5file.create_dataset("ques_len", dtype='int8', data=question_length_array)

    # h5file.create_dataset("split_train", dtype='int8', data=split)

    h5file.close()

    out_dict['ix_to_ans'] = id2ans
    out_dict['ix_to_word'] = id2word  # encode the (1-indexed) vocab
    out_dict['unique_img_train'] = unique_img

    return ans2id, word2id


def apply_vocab_question(exps, word2id):
    """
        将不在词汇表中的单词过滤，一般用于测试集
        ------------------------------------------
        Args:
        Returns:
    """
    for exp in exps:
        question = [w if w in word2id else "UNK" for w in exp["processed_tokens"]]
        exp['final_question'] = question


def val_test_split(params, out_dict, ans2id, word2id):
    exps = json.load(open(params["input_test_json"], 'r'))
    exps_val = exps[:params['val_num']]
    uni_img_val = val_or_test_data_process(exps_val, params['output_val_h5'], ans2id, word2id)
    out_dict['unique_img_val'] = uni_img_val

    exps_test = exps[params['val_num']:]
    uni_img_test = val_or_test_data_process(exps_test, params['output_test_h5'], ans2id, word2id)
    out_dict['unique_img_test'] = uni_img_test


def val_or_test_data_process(exps, h5path, ans2id, word2id):
    """
        对测试数据以及验证数据的处理：
        1. 将answer转化为索引张量
        2. 将question分词，并将不在词汇表中的词语设为UNK，将问题进行序列化
        3. 获得所有涉及到的图像数据
        4. 蜜汁split
        5. 保存结果
        ------------------------------------------
        Args:
            params: dict, the same
            h5file: h5 file, 记录张量数据的h5文件
            out_dict: dict, 记录json格式的数据
            ans2id: dict
            word2id: dict, 词汇表
        Returns:
    """
    # 1.
    ans_array = encode_answer(exps, ans2id)

    # 2.
    tokenize_questions(exps, params)
    apply_vocab_question(exps, word2id)
    question_array, question_length_array, question_id_array = encode_questions(exps, params, word2id)

    # 3.
    unique_img, img_path_array = get_unique_img(exps)

    # 4.
    # split = np.zeros(len(exps))
    # split[:] = 2

    # 5.
    h5file = h5py.File(h5path, 'w')

    h5file.create_dataset("ans", dtype='int64', data=ans_array)

    h5file.create_dataset("ques", dtype='int64', data=question_array)

    h5file.create_dataset("ques_id", dtype='int64', data=question_id_array)

    h5file.create_dataset("img_pos", dtype='int64', data=img_path_array)

    h5file.create_dataset("ques_len", dtype='int8', data=question_length_array)

    # h5file.create_dataset("split_test", dtype='int8', data=split)

    return unique_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='../data/cocoqa_raw_train.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='../data/cocoqa_raw_test.json',
                        help='input json file to process into hdf5')

    parser.add_argument('--output_json', default='../data/cocoqa_data_prepro.json', help='output json file')
    parser.add_argument('--output_train_h5', default='../data/cocoqa_prepro_train.h5', help='output h5 train file')
    parser.add_argument('--output_test_h5', default='../data/cocoqa_prepro_test.h5', help='output h5 test file')
    parser.add_argument('--output_val_h5', default='../data/cocoqa_prepro_val.h5', help='output h5 val file')

    parser.add_argument('--val_num', default=11600, type=int, help='the num of valid data')

    # options
    parser.add_argument('--max_length', default=26, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    out_dict = dict()
    out_dict['seq_len'] = params['max_length']
    a2i, w2i = train_data_process(params, out_dict)
    val_test_split(params, out_dict, a2i, w2i)
    json.dump(out_dict, open(params['output_json'], 'w', encoding='utf-8'))
