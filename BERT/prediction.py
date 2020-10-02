import pandas as pd
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from keras.layers import Lambda, Dense
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Load data
paths = {
    'test_data': "./nlp-getting-started/test.csv",
    'train_data': "./nlp-getting-started/train.csv"
}

df_test = pd.read_csv(paths['test_data'])


# Load model
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

# Make the architecture
bert = build_transformer_model(
    config_path=config['bert_config_path'],
    checkpoint_path=config['bert_checkpoint_path'],
    return_keras_model=False
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)
model = keras.models.Model(bert.model.input, output)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(config['learning_rate']),
              metrics=['sparse_categorical_accuracy'])

model.load_weights('Disaster_Rumor_Detection_best_model_1_0.weights')

# Tokenizer
tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)


# Make predictions
table = []
for idx, row in df_test.iterrows():
    token_ids, seg_ids = tokenizer.encode(row['text'], maxlen=config['max_len'])
    result = model.predict([[token_ids], [seg_ids]]).argmax(axis=1)
    table.append([row['id'], result[0]])
    print('Data id{} prediction done!'.format(row['id']))
    print('And result is {}'.format(result[0]))
    print('-'*60)

final_result = pd.DataFrame(table, columns=['id', 'target'])


if __name__ == '__main__':
    print(final_result.head())
    final_result.to_csv('mysubmission.csv', index=False)
