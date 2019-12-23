from __future__ import absolute_import, division, print_function, unicode_literals

#import all required libraries
import tensorflow as tf
import os
import pandas as pd
import matplotlib as plt

# get current working directory and Fetch input data from  '.csv' file
directory = os.getcwd() +'/input_data.csv'
df = pd.read_csv(directory)

# identify and seperate corresponding labels of features
df['label'] = pd.Categorical(df['label'])
df['label'] = df.label.cat.codes
label = df.pop('label')


# Create tendorflow datasets with feature values and labels
dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))
# print (label.values)
# for Features, gender in dataset.take(5):
#     print ('Features: {}, gender: {}'.format(Features, gender))

# Shuffle data to train algorithm and make it more robust
full_dataset = dataset.shuffle(len(df)).batch(1)

#devide Train and Test data
train_dataset = full_dataset.take(2500)
test_dataset = full_dataset.take(2500,)

# for Features, gender in train_dataset.take(5):
#     print ('Features: {}, gender: {}'.format(Features, gender))

# model
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model


model = get_compiled_model()
model.fit(train_dataset, epochs=15)

test_loss, test_acc = model.evaluate(test_dataset)
print("Tested_acc", test_acc)