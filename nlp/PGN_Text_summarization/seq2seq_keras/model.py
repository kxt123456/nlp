from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, LSTM, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

import tensorflow as tf

from pgn.batcher import train_batch_generator
from utils.data_loader import load_train_dataset
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from utils.wv_loader import Vocab, load_embedding_matrix


def seq2seq_model(input_length, output_sequence_length, embedding_matrix, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_matrix.shape[1], weights=[embedding_matrix],
                        input_length=input_length))
    model.add(Bidirectional(LSTM(256, return_sequences=False)))
    model.add(Dense(256, activation="relu"))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(1e-3))
    model.summary()
    return model


if __name__ == '__main__':
    config_gpu()
    params = get_params()

    max_enc_len = 352
    max_dec_len = 40
    batch_size = 128

    embedding_matrix = load_embedding_matrix()
    model_path = '../input/checkpoints/keras_model/epochs_2_batch_64_model.h5'
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)

    # model = seq2seq_model(max_enc_len, max_dec_len, embedding_matrix, vocab.count)

    model = tf.keras.models.load_model(model_path)

    model.fit(train_X, train_Y, batch_size=batch_size, epochs=10, use_multiprocessing=True, workers=8)
    # Save entire model to a HDF5 file
    model.save(model_path)

    # # Recreate the exact same model, including weights and optimizer.
    # model = tf.keras.models.load_model('input/epochs_10_batch_64_model.h5')
    # import numpy as np
    #
    # # 飞流直下三千尺
    # input_sen = "飞流直下三千尺"
    # char2id = [vocab.get(i, 0) for i in input_sen]
    # input_data = pad_sequences([char2id], 100)
    # result = model.predict(input_data)[0][-len(input_sen):]
    # result_label = [np.argmax(i) for i in result]
    # dict_res = {i: j for j, i in vocab.items()}
    # print([dict_res.get(i) for i in result_label])
