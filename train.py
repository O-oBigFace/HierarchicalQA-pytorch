"""
    author: W J-H (jiangh_wu@163.com)
    time: Jan 20, 2020
    -----------------------------------
    主运行函数
"""
import os
from os.path import join, exists
from tqdm import tqdm
from datetime import datetime
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser
from misc.my_dataset import MYDataset
from misc.vqa_model import VQAModel
from torch.utils.data import DataLoader



def args_process():
    arg_parser = ArgumentParser()
    # 图片
    arg_parser.add_argument("--input_img_train_h5",
                            default='data/cocoqa_data_img_vgg_train.h5',
                            help="path to the h5file containing the image feature")
    arg_parser.add_argument("--input_img_test_h5",
                            default='data/cocoqa_data_img_vgg_test.h5',
                            help="path to the h5file containing the image feature")
    arg_parser.add_argument("--input_img_val_h5",
                            default='data/cocoqa_data_img_vgg_val.h5',
                            help="path to the h5file containing the image feature")

    # 样例
    arg_parser.add_argument("--input_prepro_train_h5",
                            default='data/cocoqa_prepro_train.h5',
                            help="path to the h5file containing the preprocessed dataset")
    arg_parser.add_argument("--input_prepro_test_h5",
                            default='data/cocoqa_prepro_test.h5',
                            help="path to the h5file containing the preprocessed dataset")
    arg_parser.add_argument("--input_prepro_val_h5",
                            default='data/cocoqa_prepro_val.h5',
                            help="path to the h5file containing the preprocessed dataset")

    # 处理信息
    arg_parser.add_argument("--input_json",
                            default='data/cocoqa_data_prepro.json',
                            help="path to the json file containing additional info and vocab")

    arg_parser.add_argument("--start_from",
                            default='',
                            help="path to a model checkpoint to initialize model weights from. Empty = don't")
    arg_parser.add_argument("--co_atten_type",
                            default='Alternating',
                            help="co_attention type. Parallel or Alternating")
    arg_parser.add_argument("--feature_type",
                            default='VGG',
                            help="VGG or Residual")
    arg_parser.add_argument("--dropout",
                            default=0.5,
                            type=float,
                            help="dropout")
    arg_parser.add_argument("--hidden_size",
                            default=512,
                            type=int,
                            help="the hidden layer size of the model.")
    arg_parser.add_argument("--rnn_size",
                            default=512,
                            type=int,
                            help="size of the rnn in number of hidden nodes in each layer")
    arg_parser.add_argument("--batch_size",
                            default=128,
                            type=int,
                            help="what is the utils batch size in number of images per batch?"
                                 " (there will be x seq_per_img sentences)")
    arg_parser.add_argument("--output_size",
                            default=1000,
                            type=int,
                            help="number of output answers")
    arg_parser.add_argument("--rnn_layers",
                            default=2,
                            type=int,
                            help="number of the rnn layer")
    arg_parser.add_argument("--learning_rate",
                            default=1e-3,
                            type=float,
                            help="learning rate")
    arg_parser.add_argument("--learning_rate_decay_start",
                            default=10,
                            type=int,
                            help="at what epoch to start decaying learning rate? (-1 = dont)")
    arg_parser.add_argument("--learning_rate_decay_every",
                            default=10,
                            type=int,
                            help="every how many epoch there after to drop LR by 0.1?")
    arg_parser.add_argument("--max_epoch",
                            default=100,
                            type=int,
                            help="max number of iterations to run for (-1 = run forever)")
    arg_parser.add_argument("--save_dir",
                            default='save',
                            help="folder to save checkpoints into (empty = this folder)")
    arg_parser.add_argument("--id",
                            default=datetime.now().strftime('%y%m%d%H%M%S'),
                            help="an id identifying this run/job. "
                                 "used in cross-val and appended when writing progress files")
    arg_parser.add_argument("--seed",
                            default=123,
                            type=int,
                            help="random number generator seed to use")

    return arg_parser.parse_args().__dict__


def train(model, dataloader, device, criterion, optimizer, loss_history, acc_history):
    running_loss = 0.0
    acc_counter = 0
    counter = 0
    # iters
    print("training...")
    for i, data in tqdm(enumerate(dataloader)):
        # 数据迁移
        questions = data['question'].to(device)
        images = data['image'].to(device)
        answers = data['answer'].to(device)

        # 反向传播
        optimizer.zero_grad()
        outputs = model(questions, images)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        # 保存loss
        running_loss += loss.item()

        # 计算正确样例个数
        _, predicted = torch.max(outputs.data, 1)
        acc_counter += (predicted == answers).sum().item()
        counter += questions.shape[0]

    # save the loss
    running_loss = running_loss / len(dataloader)
    loss_history.append(running_loss)
    acc_counter = acc_counter / counter
    acc_history.append(acc_counter)


def eval(model, dataloader, device, criterion, loss_history, acc_history):
    running_loss = 0.0
    acc_counter = 0
    counter = 0
    with torch.no_grad():
        print("validing...")
        for i, data in tqdm(enumerate(dataloader)):
            questions = data['question'].to(device)
            images = data['image'].to(device)
            answers = data['answer'].to(device)

            outputs = model(questions, images)
            loss_val = criterion(outputs, answers)
            running_loss += loss_val.item()

            _, predicted = torch.max(outputs.data, 1)
            acc_counter += (predicted == answers).sum().item()
            counter += questions.shape[0]

        running_loss = running_loss / len(dataloader)
        loss_history.append(running_loss)
        acc_counter = acc_counter / counter
        acc_history.append(acc_counter)


def main(opt):
    with open(opt['input_json'], 'r') as f:
        prepro_info = json.load(f)

    opt["vocab_size"] = len(prepro_info["ix_to_word"])
    opt["output_size"] = len(prepro_info["ix_to_ans"])
    opt["seq_length"] = prepro_info["seq_len"]

    batch_size = opt['batch_size']

    dataset_train = MYDataset(path_img_h5=opt["input_img_train_h5"],
                              path_prepro_h5=opt["input_prepro_train_h5"],
                              feat_type=opt["feature_type"])
    dataset_test = MYDataset(path_img_h5=opt["input_img_test_h5"],
                             path_prepro_h5=opt["input_prepro_test_h5"],
                             feat_type=opt["feature_type"])
    dataset_val = MYDataset(path_img_h5=opt["input_img_val_h5"],
                            path_prepro_h5=opt["input_prepro_val_h5"],
                            feat_type=opt["feature_type"])
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = VQAModel(opt=opt)

    # 多GPU执行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, output_device=device)
        print(f"The model will use {torch.cuda.device_count()} GPUs!")

    model.to(device)

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=opt['learning_rate'])

    # 学习率衰减
    scheduler = StepLR(optimizer=optimizer,
                       step_size=opt["learning_rate_decay_every"],
                       last_epoch=opt["learning_rate_decay_start"]-1,
                       gamma=0.1)

    epoches = opt['max_epoch']

    loss_history, acc_history = list(), list()
    loss_history_val, acc_history_val = list(), list()
    loss_history_test, acc_history_test = list(), list()

    max_acc_val = 0

    for epoch in range(epoches):
        # train
        train(model, loader_train, device, criterion, optimizer, loss_history, acc_history)

        # valid
        eval(model, loader_val, device, criterion, loss_history_val, acc_history_val)

        # test
        eval(model, loader_test, device, criterion, loss_history_test, acc_history_test)

        scheduler.step(epoch)
        print(f"EPOCH{epoch}: ({loss_history[-1]},{acc_history[-1]}),",
              f"({loss_history_val[-1]},{acc_history_val[-1]}),",
              f"({loss_history_test[-1]}, {acc_history_test[-1]})")

        # save the model
        torch.save(model.state_dict(), join(opt['save_dir'], f'model{opt["id"]}.pkl'))

        # early stop
        if max_acc_val < acc_history_val[-1]:
            max_acc_val = acc_history_val[-1]
            torch.save(model.state_dict(), join(opt['save_dir'], f'best_model{opt["id"]}.pkl'))
            print("The best!")

        # 保存训练记录
        with open(join(opt['save_dir'], f'history{opt["id"]}'), 'w') as f:
            json.dump({
                'train_loss': loss_history,
                'train_acc': acc_history,
                'val_loss': loss_history_val,
                'val_acc': acc_history_val,
                'test_loss': loss_history_test,
                'test_acc': acc_history_test
            }, f)


if __name__ == '__main__':
    opt = args_process()
    print(json.dumps(opt, indent=2))
    torch.manual_seed = opt['seed']

    if not exists(opt['save_dir']):
        os.makedirs(opt['save_dir'])

    main(opt=opt)
