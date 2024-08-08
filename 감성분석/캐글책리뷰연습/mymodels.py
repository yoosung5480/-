import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense

# 목적

class MyModels():

    def __init__(self):
        self.layerslist
        self.vocab_size
        self.input_dim_list
        self.node_num_list
        self.str_sequense_length_list
        self.models

    def get_models(self, models):
        self.models = models
    
    def get_models_from_files(self, filename_list):



def compile_train_save(models, epochs_num, train_data, valid_data, test_data):
    histories = []
    test_results = []

    for i, model in enumerate(models):
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        history = model.fit(train_data, validation_data=valid_data, epochs=epochs_num)
        histories.append(history.history)

        filename = 'model' + str(i) + '.h5'
        model.save(filename,   # 파일이름
                   overwrite=True,         
                   include_optimizer=True, 
                   save_format='h5')
        
        test_result = model.evaluate(test_data)
        test_results.append(test_result)

    return histories, test_results
            
    
def load_models_from_files(directory, prefix='model', suffix='.h5', num_models=None):
    model_list = []
    for i in range(num_models):
        filename = f"{directory}/{prefix}{i}{suffix}"
        if os.path.exists(filename):
            model = tf.keras.models.load_model(filename)
            model_list.append(model)
        else:
            print(f"File {filename} does not exist.")
    return model_list
        


def make_models(vocab_size, node_nums, embedding_dims, layer_nums):
    vocab_size = 5000  # 예시를 위해 설정한 값, 실제 vocab_size에 맞게 변경 필요
    node_nums = [5, 10, 15]
    embedding_dims = [100, 150, 200]
    layer_nums = [1, 2, 3]

    # 모델 리스트 초기화
    model_lists = []

    # 모델 생성
    for node_num in node_nums:
        for embedding_dim in embedding_dims:
            for layer_num in layer_nums:
                layers = [
                    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed-layer'),
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(node_num, name='lstm-layer')),
                ]
                for _ in range(layer_num):
                    layers.append(tf.keras.layers.Dense(node_num, activation='relu'))
                layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

                # Sequential 모델에 레이어 추가
                model = tf.keras.Sequential(layers)
                model_lists.append(model)

    return model_lists
    # 모델 리스트 출력 (옵션)
    # for i, model in enumerate(model_lists):
    #     print(f"Model {i+1} Summary:")
    #     model.summary()
    #     print("\n")

# vocab_size = 5000  # 예시를 위해 설정한 값, 실제 vocab_size에 맞게 변경 필요
# node_nums = [5, 10, 15]
# embedding_dims = [100, 150, 200]
# layer_nums = [1, 2, 3]

# models = make_models(vocab_size, node_nums, embedding_dims, layer_nums)
# print(models)