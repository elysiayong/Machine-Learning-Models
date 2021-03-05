import numpy as np
import numpy.random as rnd
import pickle 
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import sklearn.discriminant_analysis as dis
import sklearn.naive_bayes as nbay
import sklearn.neural_network as nn
import sklearn.utils as utls 
import scipy.stats as sta


with open("linreg.pickle","rb") as file:
    dataTrain, dataTest = pickle.load(file)
    

def mean_sq_error(pred, actual):
    sq_err = np.square(actual - pred)

    return np.mean(sq_err)


def feature_map(Xtrain, K, N): 
    # Returns Z where each row of Z corresponds to [1, sin(x), ..... ] for a single x
    K_array = np.arange(1, K + 1)
    Z = np.ones(((2*K) + 1, N))

    sin_arrays = np.sin(K_array.reshape((K, 1)) * Xtrain.reshape((1, N)))
    cos_arrays = np.cos(K_array.reshape((K, 1)) * Xtrain.reshape((1, N)))

    Z[1:K+1, :] = sin_arrays
    Z[K+1:, :] = cos_arrays

    return Z.T


def plot(Xtrain, Ttrain, xvals, pred):
    plt.scatter(Xtrain, Ttrain)
    plt.plot(xvals, pred, "r")
    plt.ylim(top=max(Ttrain)+5)
    plt.ylim(bottom=min(Ttrain)-5)


def calc_err(dataTrain, dataTest, K):
    Xtrain, Ttrain, Xtest, Ttest = dataTrain[0], dataTrain[1], dataTest[0], dataTest[1]

    # Feature map data points accordingly
    Z_train = feature_map(Xtrain, K, len(Xtrain))
    Z_test = feature_map(Xtest, K, len(Xtest))
    W = (np.linalg.lstsq(Z_train, Ttrain, rcond=None))[0]

    Y_train = Z_train @ W
    Y_test = Z_test @ W

    err_train = mean_sq_error(Y_train, Ttrain)
    err_test = mean_sq_error(Y_test, Ttest)

    return W, err_train, err_test


def fit_plot(dataTrain, dataTest, K):
    Xtrain, Ttrain = dataTrain[0], dataTrain[1]

    w, err_train, err_test = calc_err(dataTrain, dataTest, K)

    xvals = np.linspace(min(Xtrain), max(Xtrain), 1000, endpoint=True)
    x_map = feature_map(xvals, K, len(xvals))
    pred_xvals = x_map @ w
    
    plot(Xtrain, Ttrain, xvals, pred_xvals)

    return w, err_train, err_test


def show_outputs(dataTrain, dataTest, title, K):
    w, err_train, err_test = fit_plot(dataTrain, dataTest, K)
    plt.suptitle(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    print("K-value: {0}".format(K))
    print("Training Error: {0}".format(err_train))
    print("Test Error: {0}".format(err_test))
    print("Weight: {0}".format(w))


def plot12(dataTrain, title):
    xvals = np.linspace(min(dataTrain[0]), max(dataTrain[0]), 1000, endpoint=True)
    plt.figure()

    for i in range(1, 13):
        plt.subplot(4, 3, i)
        fit_plot(dataTrain, dataTest, i)

    plt.suptitle(title)
    plt.show()


def kfolds():
    # Divide dataTrain into 5 equal folds
    dataFolds = np.array([dataTrain[:, :5], dataTrain[:, 5:10], dataTrain[:, 10:15], dataTrain[:, 15:20], dataTrain[:, 20:25]])

    avg_train_err_ls = []
    avg_val_err_ls = []

    for k in range(13):
        trainFold = np.ones((2, 20))
        train_err_ls = []
        val_err_ls = []

        for i in range(len(dataFolds)):
            validFold = dataFolds[i]
            # Add folds accordingly
            trainFold[0, :] = np.append(dataFolds[0:i, 0], dataFolds[i+1:, 0])
            trainFold[1, :] = np.append(dataFolds[0:i, 1], dataFolds[i+1:, 1])

            w, err_train, err_val = calc_err(trainFold, validFold, k)
            
            train_err_ls.append(err_train)
            val_err_ls.append(err_val)
        
        avg_train_err_ls.append(np.mean(train_err_ls))
        avg_val_err_ls.append(np.mean(val_err_ls))


    plt.figure()
    plt.semilogy(range(13), avg_train_err_ls, "b")
    plt.semilogy(range(13), avg_val_err_ls, "r")
    plt.xlabel("K")
    plt.ylabel("Mean error")
    plt.suptitle("Mean training and validation error")
    plt.show()

    min_K = np.argmin(avg_val_err_ls)
    show_outputs(dataTrain, dataTest, "The best fitting function", min_K)


# Sample scripts to run

# Show fitted function where the K-value is 12
show_outputs(dataTrain, dataTest, "The fitted function (K=12)", 12)

# Plot fitted functions from K=1 to K=12
plot12(dataTrain, "Fitted functions for different values of K")

# Train model on 5 folds
kfolds()