import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def sigmoid(scores):
   return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
   scores = np.dot(features, weights)
   ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
   return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in xrange(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print log_likelihood(features, target, weights)
    return weights


def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index))

def shuffle_file(outputfilename):
  shuffler(os.getcwd() +'/input_data.csv').to_csv(outputfilename, sep=',', index=False)


# ------------  Read Data from file -----------------------
directory = os.getcwd() +'/final-output.csv'
shuffle_file(directory)
input_data = pd.read_csv(directory)


# -----------------identify and seperate corresponding labels of features-----------------
input_data['label'] = pd.Categorical(input_data['label'])
input_data['label'] = input_data.label.cat.codes
label = input_data.pop('label')

# -----------------Train dataset for Training evaluation-----------------------
train_data = input_data.values[:2500, :]
test_data = np.zeros(input_data.shape[0]- 2500)
test_data = input_data.values[2500:, :]
train_label = np.zeros(2500)

# -----------------Test dataset for evaluation------------------------
test_label = np.zeros(input_data.shape[0]- 2500)
train_label = label[:2500]
test_label = label[2500:]


# -----------------Training logistic regression model for evaluation------------------------
weights = logistic_regression(train_data, train_label,
                     num_steps = 300000, learning_rate = 6e-6, add_intercept=True)

# --------------train dataset Evaluation-----------------

data_with_intercept = np.hstack((np.ones((train_data.shape[0], 1)),
                                 train_data))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print 'Train Accuracy: {0}'.format((preds == train_label).sum().astype(float) / len(preds))

# ------------Test dataset Evaluation-------------


data_with_intercept = np.hstack((np.ones((test_data.shape[0], 1)),
                                 test_data))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print 'Test Accuracy: {0}'.format((preds == test_label).sum().astype(float) / len(preds))
