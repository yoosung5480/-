import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from collections import Counter
# 데이터 분석은 스스로해서 데이터가 아래와 같은 형식이라고 가정하에 진행할것이다.
# index : review : label
# 토큰을 뽑아서, 특정 토큰의 빈도나, 리뷰 점수에 따른 라벨링은 다른곳에서 스스로 진행해야한다.


# 목표 : 맥도날드 리뷰 분류(영어 리뷰 기준)
# 수행할것
# 1. 데이터 전처리P
#   분할 (테스트, 검증, 훈련)
#   토큰화

# 2. 모델 훈련 
#   파라미터에 따라 모델 훈련
#   모델 훈련 결과 저장

# 3. 앙상블 학습 실행


# 데이터 전처리 및 토큰화
class PreProcess ():
#---------------------------------------------------------------------------#
    def __init__(self, ds_raw, target_size, is_tokenize = False):
        self.ds_raw = ds_raw
        self.target_size = target_size
        self.is_tokenize = is_tokenize
#---------------------------------------------------------------------------#
    # 데이터셋 분리, 분리하기전에 무작위로 샘플끼리 섞고, 다시 섞지 않은채 데이터를 비율에 맞추서 뽑고, 반환한다.
    # 분리 메서드함수
    def _divide(self, train_size=0.4, valid_size = 0.1, test_size=0.5):
        ds_raw = self.ds_raw
        assert train_size + valid_size + test_size == 1.0, "The sum of train, valid and test sizes must be 1.0"
        train_num = train_size * self.target_size
        test_num = test_size * self.target_size
        valid_num = valid_size * self.target_size

        tf.random.set_seed(1)
        ds_raw = ds_raw.shuffle(self.target_size, reshuffle_each_iteration=False)

        self.ds_raw_test = ds_raw.take(int(train_num))
        self.ds_raw_train = ds_raw.take(int(test_num))
        self.ds_raw_valid = ds_raw.take(int(valid_num))

        return self.ds_raw_train, self.ds_raw_valid, self.ds_raw_test
    
#---------------------------------------------------------------------------#

#---------------------------------------------------------------------------#
    # _encode -> _encode_map_fn -> tokenize
    # 토근화, 모델 훈련전에 반드시 거쳐야한다.
    def _tokenize(self) :
        if self.is_tokenize == False:
            self.tokenizer = tfds.deprecated.text.Tokenizer()
            self.token_counts = Counter()

            for ex in self.ds_raw:
                tokens = self.tokenizer.tokenize(ex[0].numpy()[0])
                self.token_counts.update(tokens) 

            self.encoder = tfds.deprecated.text.TokenTextEncoder(self.token_counts)

            self.ds_train = self.ds_raw_train.map(self._encode_map_fn)
            self.ds_valid = self.ds_raw_valid.map(self._encode_map_fn)
            self.ds_test = self.ds_raw_test.map(self._encode_map_fn)
            self.is_tokenisze = True
            return self.ds_train, self.ds_valid, self.ds_test
        
        else:
            self.ds_train = self.ds_raw_train
            self.ds_valid = self.ds_raw_valid
            self.ds_test = self.ds_raw_test
            return self.ds_train, self.ds_valid, self.ds_test

    # 텍스트 변환 인코딩 함수 정의 (텍스트_텐서)
    def _encode(self,text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = self.encoder.encode(text)
        return encoded_text, label


    # 함수를 TF연산으로 변환하기
    def _encode_map_fn(self, text, label):
        return tf.py_function(self._encode, inp=[text, label], Tout=(tf.int64, tf.int64))
#---------------------------------------------------------------------------#

    def get_datas(self, batch_size, train_size=0.4, valid_size = 0.1, test_size=0.5):
        self._divide(train_size, valid_size, test_size)
        self._tokenize()

        train_data = self.ds_train.padded_batch(batch_size, padded_shapes=([-1], []))
        valid_data = self.ds_valid.padded_batch(batch_size, padded_shapes=([-1], []))
        test_data = self.ds_test.padded_batch(batch_size, padded_shapes=([-1], []))
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        return train_data, valid_data, test_data
#---------------------------------------------------------------------------#

def preprocessing_tokinize(df_str, max_pad_len = 100):
    # Keras Tokenizer를 사용한 토큰화 및 시퀀스 변환
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_str)
    sequences = tokenizer.texts_to_sequences(df_str)
    padded_sequences = pad_sequences(sequences, maxlen=max_pad_len)
    token_counts = Counter(tokenizer.word_counts)
    return padded_sequences, token_counts

def compile_train_save(models, epochs_num, train_data, valid_data, test_data):
    histories = []
    test_results = []

    for i, model in enumerate(models):
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        history = model.fit(train_data, validation_data=valid_data, epochs=epochs_num)
        histories.append(history.history)

        filename = 'models/' + 'model' + str(i) + '.h5'
        model.save(filename,   # 파일이름
                   overwrite=True,         
                   include_optimizer=True, 
                   save_format='h5')
        
        test_result = model.evaluate(test_data)
        test_results.append(test_result)

    return histories, test_results
            
    
def load_models_from_files(directory='/Users/yujin/Desktop/python_projects/머신러닝교과서/감성분석/models', prefix='model', suffix='.h5', num_models=None):
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
    # vocab_size = 5000  # 예시를 위해 설정한 값, 실제 vocab_size에 맞게 변경 필요
    # node_nums = [5, 10, 15]
    # embedding_dims = [100, 150, 200]
    # layer_nums = [1, 2, 3]

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


def majority_vote_ensemble(model_list, test_data, accuracy=0.5):
    # 각 모델의 예측 결과를 저장할 리스트
    all_predictions = []

    # 각 모델의 예측 결과를 수집
    for model in model_list:
        predictions = model.predict(test_data)
        # 예측 결과를 0 또는 1로 변환 (0.5를 기준으로)
        binary_predictions = (predictions > accuracy).astype(int)
        all_predictions.append(binary_predictions)

    # 다수결 투표를 위해 각 모델의 예측 결과를 모음
    all_predictions = np.array(all_predictions)  # Shape: (num_models, num_samples, 1)
    all_predictions = np.squeeze(all_predictions, axis=-1)  # Shape: (num_models, num_samples)

    # 다수결 투표
    majority_vote_predictions = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)),
        axis=0,
        arr=all_predictions
    )

    return majority_vote_predictions


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
