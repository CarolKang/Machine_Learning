from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt

"""
   Featurized least square, gradient descent and stochastic gradient descent.
"""

NUM_CLASSES = 10
d = 8222 #8000
sigma = 0.15 #0.13
G = None
b = None

gd_x = []
gd_y = []
sgd_x = []
sgd_y = []


def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)

def loss(W, X, Y, reg):
    retval = 0
    for i in range(X.shape[0]):
        retval += np.linalg.norm(np.matmul(W.T, X[i].T) - Y[i].T)**2
    return retval + reg * (np.linalg.norm(W) ** 2)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    y_sum = np.dot(np.matrix(X_train).T, np.matrix(y_train))
    x_sum = np.dot(np.matrix(X_train).T, np.matrix(X_train))

    return np.dot(np.linalg.inv(x_sum + reg * np.identity(X_train.shape[1])), y_sum)
    


def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    global gd_x, gd_y
    XTX = np.matmul(X_train.T, X_train)
    XTy = np.matmul(X_train.T, y_train)
    converged = False
    i = 0
    W = np.random.random((X_train.shape[1], NUM_CLASSES))
    while (i < num_iter and converged == False):
        delta = (alpha / X_train.shape[0]) * 2 * (np.matmul(XTX, W) - XTy + reg * W)
        W = W - delta
        if not np.any(delta):
            converged = True
        i += 1

        # ---- Plotting ----
        if (i % 100 == 0):
            l = loss(W, X_train, y_train, reg)
            gd_x.append(i)
            gd_y.append(l)

    return W

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    global sgd_x, sgd_y
    W = np.random.random((X_train.shape[1], NUM_CLASSES))
    converged = False
    i = 0
    while (i < num_iter and converged == False):
        idx = np.random.randint(X_train.shape[0])
        Xi = np.matrix(X_train[idx]).T
        yiT = np.matrix(y_train[idx])
        delta = alpha / (1 + alpha * reg * i) * 2 * ((np.matmul(Xi, (np.matmul(Xi.T, W)) - yiT)) + reg * W)
        W = W - delta
        if not np.any(delta):
            converged = True
        i += 1
        # ---- Plotting ----
        if (i % 1000 == 0):
            l = loss(W, X_train, y_train, reg)
            sgd_x.append(i)
            sgd_y.append(l)

    return W

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    retMatrix = []
    for label in labels_train:
        toApp = np.zeros(NUM_CLASSES)
        toApp[label] = 1
        retMatrix.append(toApp)
    return np.array(retMatrix)

def predict(model, X):
    result = np.dot(np.matrix(model).T, np.matrix(X).T).T
    return [np.argmax(result[i]) for i in range(result.shape[0])]
    # return [np.argmax(np.matmul(model.T, x)) for x in X]

def phi(X):
    ''' Featurize the inputs using random Fourier features '''
    # original shape, lifted shape and sigma
    global G, b
    if G is None:
        p = X.shape[1]
        G = np.random.normal(0, sigma, (p, d))
        b = np.random.uniform(0, 2*np.pi, (d, 1))
    B = np.tile(b, (1, X.shape[0])).T
    return np.cos(np.matmul(X,G) + B)




if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    X_train, X_test = phi(X_train), phi(X_test)

    model = train(X_train, y_train, reg=0.1)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)

    # --- for Kaggle csv Submission ---
    with open('wenjing.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id','Category'])
        for i in range(len(pred_labels_test)):
            writer.writerow([i, pred_labels_test[i]])

    print("Closed form solution")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=30000)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Batch gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # model = train_sgd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=300000)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Stochastic gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # plt.figure(1)
    # plt.title('Gradient Descent')
    # plt.xlabel('Iterations')
    # plt.ylabel('Training Error')
    # plt.semilogy(gd_x, gd_y)
    # plt.show()

    # plt.figure(2)
    # plt.title('Stochastic Gradient Descent')
    # plt.xlabel('Iterations')
    # plt.ylabel('Training Error')
    # plt.semilogy(sgd_x, sgd_y)
    # plt.show()


