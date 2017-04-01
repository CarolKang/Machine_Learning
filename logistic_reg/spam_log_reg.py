import sklearn.metrics as metrics
import numpy as np
import scipy
from scipy import io 
from scipy import linalg as la
import csv
import matplotlib.pyplot as plt

"""
    Spam detection using logistic regression
"""

gd_x = []
gd_y = []
sgd_x = []
sgd_y = []


def load_dataset():
    data = scipy.io.loadmat('data/spam.mat')
    X_train = data['Xtrain']
    Y_train = data['ytrain']
    X_test = data['Xtest']

    return X_train, Y_train, X_test

def sigmoid(X_train, beta):
    return 1 / (1 + np.exp(- np.dot(X_train, beta)))

def standardize(X_train):
    # print(X_train.shape)
    for i in range(X_train.shape[1]):
        col = X_train[:,i]
        mean = np.mean(col)
        std = np.std(col)
        X_train[:,i] = (col - mean)/std
        # print(X_train.shape)
    return X_train

def transform(X_train):
    # print(X_train.shape)
    # print((X_train + 0.1).shape)
    return np.log(X_train + 0.1)

def binarize(X_train):
    return np.where(X_train > 0, 1, 0)
def plot(x, y, method):
    plt.figure()
    plt.semilogy(x, y)
    plt.xlabel("# of Iterations")
    plt.ylabel("Train Error")
    plt.title(method + " Training loss")
    plt.savefig(method + ".png")

def logistic_gd(X_train, Y_train, alpha = 0.001, reg = 0.1, num_iter = 10000, step = 1):
    global gd_x, gd_y
    beta = np.random.normal(0, 0.2, (X_train.shape[1], 1))
    for i in range(num_iter):
        mu = sigmoid(X_train, beta)
        gradient = 2 * reg * beta - np.dot(X_train.T, Y_train - mu)
        beta -= alpha/X_train.shape[0] * gradient
        if i % step == 0:
            # delta = sigmoid(X_train, beta)
            pred_spam_train = predict(beta, X_train)
            accuracy = metrics.accuracy_score(Y_train, pred_spam_train)
            gd_x.append(i)
            gd_y.append(1 - accuracy) 
    return beta

def logistic_sgd(X_train, Y_train, alpha = 0.001, reg = 0.1, num_iter = 10000, step = 1):
    global sgd_x, sgd_y
    beta = np.random.normal(0,0.2,(X_train.shape[1], 1))
    for i in range(num_iter):
        idx = np.random.randint(X_train.shape[0])
        Xi = np.matrix(X_train[idx])
        yi = np.matrix(Y_train[idx])
        mu = sigmoid(Xi, beta)
        gradient = 2 * reg * beta - np.dot(Xi.T, (yi - mu))
        beta -= alpha / (1 + alpha * reg * i) * gradient
        if i % step == 0:
            pred_spam_train = predict(beta, X_train)
            accuracy = metrics.accuracy_score(Y_train, pred_spam_train)
            sgd_x.append(i)
            sgd_y.append(1 - accuracy)  
    return beta

def predict(model, X):
    return np.rint(sigmoid(X, model))

if __name__ == "__main__":
    X_train, Y_train, X_test = load_dataset()
    
    # standardize
    model = logistic_gd(standardize(X_train), Y_train, alpha = 0.15, reg = 0, num_iter = 6000, step = 1)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, standardize(X_test))
    print("Train accuracy: {0}".format(metrics.accuracy_score(Y_train, pred_labels_train)))
    plot(gd_x, gd_y, "Logistic Gradient(standardized)")
    
    model = logistic_sgd(standardize(X_train), Y_train, alpha = 0.15, reg = 0.06, num_iter = 60000, step = 1)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, standardize(X_test))
    print("Train accuracy: {0}".format(metrics.accuracy_score(Y_train, pred_labels_train)))
    plot(sgd_x, sgd_y, "Logistic Stochastic Gradient(standardized)")
    
    # transform
    model = logistic_gd(transform(X_train), Y_train, alpha = 0.13, reg = 0.08, num_iter = 100000, step = 1)
    pred_labels_train = predict(model, transform(X_train))
    pred_labels_test = predict(model, transform(X_test))
    print("Train accuracy: {0}".format(metrics.accuracy_score(Y_train, pred_labels_train)))
    plot(gd_x, gd_y, "Logistic Gradient(transformed)")
    
    model = logistic_sgd(transform(X_train), Y_train, alpha = 0.15, reg = 0.06, num_iter = 100000, step = 1)
    pred_labels_train = predict(model, transform(X_train))
    pred_labels_test = predict(model, transform(X_test))
    print("Train accuracy: {0}".format(metrics.accuracy_score(Y_train, pred_labels_train)))
    plot(sgd_x, sgd_y, "Logistic Stochastic Gradient(transformed)")

    # binarize
    model = logistic_gd(binarize(X_train), Y_train, alpha = 0.15, reg = 0.06, num_iter = 60000, step = 1)
    pred_labels_train = predict(model, binarize(X_train))
    pred_labels_test = predict(model, binarize(X_test))
    print("Train accuracy: {0}".format(metrics.accuracy_score(Y_train, pred_labels_train)))
    plot(gd_x, gd_y, "Logistic Gradient(binarized)")
    
    model = logistic_sgd(binarize(X_train), Y_train, alpha = 0.15, reg = 0.06, num_iter = 60000, step = 1)
    pred_labels_train = predict(model, binarize(X_train))
    pred_labels_test = predict(model, binarize(X_test))
    print("Train accuracy: {0}".format(metrics.accuracy_score(Y_train, pred_labels_train)))
    plot(sgd_x, sgd_y, "Logistic Stochastic Gradient(binarized)")


    # # --- for Kaggle csv Submission ---
    # with open('wenjing.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Id','Category'])
    #     for i in range(len(pred_labels_test)):
    #         writer.writerow([i+1, int(pred_labels_test[i][0])])

    


