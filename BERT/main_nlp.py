import argparse
import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import math
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#TODO Check the dataset
paths = {
    'test_data': "./nlp-getting-started/test.csv",
    'train_data': "./nlp-getting-started/train.csv"
}

df_train = pd.read_csv(paths['train_data'])
df_test = pd.read_csv(paths['test_data'])
print(df_train)



# TODO Start training the model
config = {
    'model_name': 'Disaster_Rumor_Detection',
    'model_version': '1',
    'fold': '0',
    'max_len': 512,
    'labels': [0, 1],
    'batch_size': 16,
    'epoch': 10,
    'learning_rate': 1e-5,
    'gpu_mem_fraction': 0.7,

    # You should download the following files from Google BERT research website(pre-trained models)
    'bert_config_path': '/home/hning/adversarial/limited-blackbox-attacks-master/JerryWorkFolder/uncased_L-12_H-768_A-12/bert_config.json',
    'bert_checkpoint_path': '/home/hning/adversarial/limited-blackbox-attacks-master/JerryWorkFolder/uncased_L-12_H-768_A-12/bert_model.ckpt',
    'dict_path': '/home/hning/adversarial/limited-blackbox-attacks-master/JerryWorkFolder/uncased_L-12_H-768_A-12/vocab.txt',
    'bert_layers': 6
}

tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_function = config['gpu_mem_fraction']
sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()


def load_data(df):
    D = []
    for idx, row in df.iterrows():
        text, label = row['text'], row['target']
        D.append((text, label))
    return D


# Load the dataset for training
df_train, df_val = train_test_split(df_train, test_size=.25, random_state=41)
train_data = load_data(df_train)
test_data = load_data(df_val)
print(f'how many TRAIN data: {len(train_data)}')
print(f'how many TEST data: {len(test_data)}')
print(df_val)


# Build the Tokenizer to divide the sentences
tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)


# For data generation
class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=config['max_len'])
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# Data Transformation
train_generator = data_generator(train_data, config['batch_size'])
test_generator = data_generator(test_data, config['batch_size'])


def _create_model(bert_config_path, bert_checkpoint_path, bert_layers, num_labels):
    # Load the pre-trained model
    bert = build_transformer_model(
        config_path=bert_config_path,
        checkpoint_path=bert_checkpoint_path,
        return_keras_model=False
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(units=num_labels,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)
    model = keras.models.Model(bert.model.input, output)
    return model


model = _create_model(bert_config_path=config['bert_config_path'],
                      bert_checkpoint_path=config['bert_checkpoint_path'],
                      bert_layers=config['bert_layers'],
                      num_labels=len(config['labels']))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(config['learning_rate']),
              metrics=['sparse_categorical_accuracy'])

def evaluate(data):
    """
    accuracy score
    """
    total, right = 0, 0
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        print('what is: {}'.format(x_true))
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right/total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0
        self.model_path = f'./'

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(test_generator)
        # print('saving model...')
        # model.save_weights(f"./{self.model_path}/{config['model_name']}_best_model_{epoch}.weights")
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            print('saving model ...')
            model.save(f"./{self.model_path}/{config['model_name']}_best_model_{int(config['model_version'])+1}_{config['fold']}.weights")
        print('val_acc: {:.5f}, best_val_acc: {:.5f}\n'.format(val_acc, self.best_val_acc))


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=config['epoch'],
                        callbacks=[evaluator])


