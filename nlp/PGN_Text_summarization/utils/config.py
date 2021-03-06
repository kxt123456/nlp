# -*- coding:utf-8 -*-
# Created by xiangtao.kong at 03/02/20
import os
import pathlib

# 预处理数据 构建数据集
is_build_dataset = True

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 训练数据路径
train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv').replace('\\','/')
# 测试数据路径
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv').replace('\\','/')
# 停用词路径
# stop_word_path = os.path.join(root, 'input', 'stopwords/哈工大停用词表.txt')
stop_word_path = os.path.join(root, 'data', 'stopwords/stopwords.txt').replace('\\','/')

# 自定义切词表
user_dict = os.path.join(root, 'data', 'user_dict.txt').replace('\\','/')

# 0. 预处理
# 预处理后的训练数据
train_seg_path = os.path.join(root, 'data', 'train_seg_data.csv').replace('\\','/')
# 预处理后的测试数据
test_seg_path = os.path.join(root, 'data', 'test_seg_data.csv').replace('\\','/')
# 合并训练集测试集数据
merger_seg_path = os.path.join(root, 'data', 'merged_train_test_seg_data.csv').replace('\\','/')

# 1. 数据标签分离
train_x_seg_path = os.path.join(root, 'data', 'train_X_seg_data.csv').replace('\\','/')
train_y_seg_path = os.path.join(root, 'data', 'train_Y_seg_data.csv').replace('\\','/')

val_x_seg_path = os.path.join(root, 'data', 'val_X_seg_data.csv').replace('\\','/')
val_y_seg_path = os.path.join(root, 'data', 'val_Y_seg_data.csv').replace('\\','/')

test_x_seg_path = os.path.join(root, 'data', 'test_X_seg_data.csv').replace('\\','/')

# 2. pad oov处理后的数据
train_x_pad_path = os.path.join(root, 'data', 'train_X_pad_data.csv').replace('\\','/')
train_y_pad_path = os.path.join(root, 'data', 'train_Y_pad_data.csv').replace('\\','/')
test_x_pad_path = os.path.join(root, 'data', 'test_X_pad_data.csv').replace('\\','/')

# 3. numpy 转换后的数据
train_x_path = os.path.join(root, 'data', 'train_X').replace('\\','/')
train_y_path = os.path.join(root, 'data', 'train_Y').replace('\\','/')
test_x_path = os.path.join(root, 'data', 'test_X').replace('\\','/')

# 词向量路径
save_wv_model_path = os.path.join(root, 'data', 'wv', 'word2vec.model').replace('\\','/')
# 词向量矩阵保存路径
embedding_matrix_path = os.path.join(root, 'data', 'wv', 'embedding_matrix_300').replace('\\','/')
# 字典路径
vocab_path = os.path.join(root, 'data', 'wv', 'vocab_300.txt').replace('\\','/')
reverse_vocab_path = os.path.join(root, 'data', 'wv', 'reverstest_save_dire_vocab.txt').replace('\\','/')

# 词向量训练轮数
wv_train_epochs = 10

# 模型保存文件夹
# checkpoint_dir = os.path.join(root, 'input', 'checkpoints', 'training_checkpoints_pgn_cov_not_clean')

checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_pgn_cov_backed')

# checkpoint_dir = os.path.join(root, 'input', 'checkpoints', 'training_checkpoints_seq2seq')
seq2seq_checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_seq2seq')

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 结果保存文件夹
save_result_dir = os.path.join(root, 'result')

# 词向量维度
embedding_dim = 300

sample_total = 82871

batch_size = 6

epochs = 2

vocab_size = 50000
