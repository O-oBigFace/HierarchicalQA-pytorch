"""
    author: W J-H (jiangh_wu@163.com)
    time: Jan 5, 2020
    -----------------------------------
    对cocoqa的数据进行预处理
    待完成：添加argparser
"""

from os.path import join
import json


def convert(data_dir, dataset, ftype, subtype):
    """
        将cocoqa的数据转化为格式化形式
        ------------------------------------------
        Args:
            dataset: dataset dir name
            ftype: "train" or "test"
            subtype: "train2014" or "val2014"
        Returns:
            res: list
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
    """

    with open(join(data_dir, dataset, ftype, "answers.txt"), 'r', encoding="utf-8") as f:
        answers = [line.strip() for line in f.readlines()]

    with open(join(data_dir, dataset, ftype, "img_ids.txt"), 'r', encoding="utf-8") as f:
        imgids = [int(line.strip()) for line in f.readlines()]

    with open(join(data_dir, dataset, ftype, "questions.txt"), 'r', encoding="utf-8") as f:
        questions = [line.strip() + ' ?' for line in f.readlines()]

    with open(join(data_dir, dataset, ftype, "types.txt"), 'r', encoding="utf-8") as f:
        types = [line.strip() for line in f.readlines()]

    assert len(answers) == len(imgids) == len(questions) == len(types)

    res = list()
    N = len(answers)
    for i in range(N):
        res.append({
            'ques_id': i,
            'img_path': "%s/COCO_%s_%012d.jpg" % (subtype, subtype, imgids[i]),
            'question': questions[i],
            'ans': answers[i],
            'types': types[i]
        })

    return res


if __name__ == '__main__':
    data_dir = "../data"
    dataset = "cocoqa-2015-05-17"
    PATH_RAW_TRAIN = "../data/cocoqa_raw_train.json"
    PATH_RAW_TEST = "../data/cocoqa_raw_test.json"

    train_data = convert(data_dir, dataset, "train", "train2014")
    json.dump(train_data, open(PATH_RAW_TRAIN, "w"), indent=2)
    test_data = convert(data_dir, dataset, "test", "val2014")
    json.dump(test_data, open(PATH_RAW_TEST, "w"), indent=2)

