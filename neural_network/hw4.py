from mnist import MNIST
import numpy as np
import sklearn.metrics as metrics
import numpy as np
import scipy
from scipy import io 
from scipy import linalg as la
import csv
import matplotlib.pyplot as plt
"""
    Digit recognition using three-layer Neural Network.
"""
n_in = 784
n_hid = 500

NUM_CLASSES = 10
sgd_x = []
sgd_y = []
sgd_err = []


def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_test, _ = map(np.array, mndata.load_testing())
    # Remember to center and normalize the data...
    return X_train, labels_train, X_test

def plot(x, y, msg):
    plt.figure()
    plt.semilogy(x, y)
    plt.xlabel("# of Iterations")
    plt.ylabel("Train " + msg)
    plt.title("Training " + msg)
    # plt.savefig(method + ".png")

def sigmoid(X):
    return 1 / (1 + np.exp(- X))

def dSigmoid(X):
	mu = sigmoid(X)
	return mu * (1 - mu)

def preprocess(X_train):
    mean = X_train.mean()
    std = X_train.std()
    return (X_train - mean) / std

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.eye(NUM_CLASSES)[labels_train]

def softmax(X):
	'''Numerically stable way for softmax'''
	new_X = X - np.max(X)
	return np.exp(new_X) / np.sum(np.exp(new_X))

def relu(X):
	return np.maximum(X, 0)

def dRELU(X):
	X[X>0] = 1
	return X


def train_sgd(X_train, Y_train, alpha = 0.001, reg = 0.1, num_iter = 10000, step = 1):
    global sgd_x, sgd_y, sgd_err
    V = np.random.normal(0,0.01,(n_hid, n_in + 1))
    W = np.random.normal(0,0.01,(NUM_CLASSES,n_hid + 1))
    for i in range(num_iter):
        idx = np.random.randint(X_train.shape[0])
        Xi = np.matrix(X_train[idx])
        Xi = np.insert(Xi, 0, 1)
        yi = np.matrix(Y_train[idx])

        S_h = np.dot(Xi, V.T)
        H = relu(S_h)
        X_H = np.insert(H, 0, 1)

        S_o = np.dot(X_H, W.T)
        X_O = softmax(S_o)


        dW = np.dot((X_O - yi).T, X_H)

        W_h = W[:,:-1]
        dV = np.dot(np.dot((X_O - yi), W_h)* dRELU(S_h).T, Xi)

        decay = alpha / (1 + alpha * reg * i)

        prev_W = W
        prev_V = V

        W = W - decay * dW
        V = V - decay * dV

        if i % step == 0:
        	pred_labels_train = predict(V, W, X_train)
        	accuracy = metrics.accuracy_score(labels_train, pred_labels_train)
            # if (len(sgd_y) > 0) and (accuracy < sgd_y[-1]):
            #     return prev_V, prev_W
        	sgd_x.append(i)
        	sgd_y.append(accuracy)  
        	sgd_err.append(1 - accuracy)
    return V, W

def predict(V, W, X):
	''' From model and data points, output prediction vectors '''
	X = np.insert(X, 0, 1, axis=1)
	S_h = np.dot(X, V.T)
	H = relu(S_h)
	X_H = np.insert(H, 0, 1, axis=1)

	S_o = np.dot(X_H, W.T)
	X_O = softmax(S_o)
	return np.argmax(X_O, axis=1)

	


if __name__ == "__main__":
    X_train, labels_train, X_test = load_dataset()
    Y_train = one_hot(labels_train)
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    idxs = np.random.choice(60000, 10000, replace = False)
    new_Test = X_train[idxs, :]
    new_Train = X_train[np.setdiff1d(np.arange(60000), idxs), :]
    new_Y_test = Y_train[idxs, :]
    new_Y_train = Y_train[np.setdiff1d(np.arange(60000), idxs), :]

    import time
    start_time = time.time()
    # V, W = train_sgd(new_Train, new_Y_train, alpha = 0.001, reg = 0.1, num_iter = 10000, step = 1)
    V, W = train_sgd(X_train, Y_train, alpha = 0.001, reg = 0.1, num_iter = 100, step = 1)
    print("--- %s seconds ---" % (time.time() - start_time))

    # pred_labels_train = predict(V, W, new_Train)
    # pred_labels_test = predict(V, W, new_Test)
    # train_score = metrics.accuracy_score(new_Y_train, pred_labels_train)
    # test_score = metrics.accuracy_score(new_Y_test, pred_labels_test)
    # print("Train accuracy: {0}".format(train_score))
    # print("Test accuracy: {0}".format(validation_score)")

    pred_labels_train = predict(V, W, X_train)
    train_score = metrics.accuracy_score(labels_train, pred_labels_train)
    print("Train accuracy: {0}".format(train_score))
    
    plot(sgd_x, sgd_y, "accuracy")
    plot(sgd_x, sgd_err, "error")

    with open('wenjing.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id','Category'])
        for i in range(len(pred_labels_train)):
            writer.writerow([i+1, int(pred_labels_train[i][0])])


