import numpy as np
import numpy.random as rnd
import pickle 
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis as dis
import sklearn.naive_bayes as nbay
import sklearn.utils as utls 
import scipy.stats as sta


def plot_data(X, T):
    # Separate class by labels 
    class0, class1, class2 = X[T == 0, :], X[T == 1, :], X[T == 2, :]

    plt.scatter(class0[:, 0], class0[:, 1], s=2, c='r')
    plt.scatter(class1[:, 0], class1[:, 1], s=2, c='b')
    plt.scatter(class2[:, 0], class2[:, 1], s=2, c='g')

    plt.xlim(min(X[:, 0]-0.1), max(X[:, 0]+0.1))
    plt.ylim(min(X[:, 1]-0.1), max(X[:, 1]+0.1))

def print_accuracy(accuracy1, accuracy2):
    print("sklearn accuracy: {}".format(accuracy1))
    print("peronal model accuracy: {}".format(accuracy2))
    print("accuracy difference: {}".format(abs(accuracy1 - accuracy2)))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)


def accuracyQDA(clf, X, T):
    mean = clf.means_
    cov = clf.covariance_
    prior_probs = clf.priors_

    class_pdf_matrix = np.empty((np.shape(mean)[0], np.shape(X)[0]), float)
    for i in range(np.shape(mean)[0]):
        class_pdf = sta.multivariate_normal.pdf(X, mean[i], cov[i])
        class_pdf_matrix[i] = class_pdf

    class_pdf_matrix = class_pdf_matrix.T
    pred = np.log(prior_probs) + np.log(class_pdf_matrix)
    accuracy = np.mean(np.argmax(pred, axis=len(np.shape(pred))-1) == T)

    return accuracy


def accuracyNB(clf, X, T):
        mean = clf.theta_
        var = clf.sigma_
        prior_probs = clf.class_prior_

        new_X = X[:, np.newaxis]
        exp_t = -1 * (new_X - mean)**2 / (2 * var)
        div = np.sqrt(2 * np.pi * var)

        prob = np.exp(exp_t) / div
        pred = np.log(prior_probs) + np.log(np.prod(prob, axis=len(np.shape(prob))-1)) 

        accuracy = np.mean(np.argmax(pred, axis=len(np.shape(pred))-1) == T)
        return accuracy


def script_qda():
    with open("./data/data2.pickle","rb") as file:
        dataTrain, dataTest = pickle.load(file)
    Xtrain, Ttrain = dataTrain
    Xtest, Ttest = dataTest

    clf = dis.QuadraticDiscriminantAnalysis(store_covariance=True)
    clf.fit(Xtrain, Ttrain)

    #accuracy calculated using sklearn
    accuracy1 = clf.score(Xtest, Ttest)
    accuracy2 = accuracyQDA(clf, Xtest, Ttest)

    print_accuracy(accuracy1, accuracy2)


    plot_data(Xtrain, Ttrain)
    plt.suptitle("Decision boundaries for quadratic discriminant analysis")
    plt.show()

def script_naive_bayes():
    with open("./data/data2.pickle","rb") as file:
        dataTrain, dataTest = pickle.load(file)
    Xtrain, Ttrain = dataTrain
    Xtest, Ttest = dataTest

    clf = nbay.GaussianNB()
    clf.fit(Xtrain, Ttrain)

    #accuracy calculated using sklearn
    accuracy1 = clf.score(Xtest, Ttest)
    #model accuracy
    accuracy2 = accuracyNB(clf, Xtest, Ttest)

    print_accuracy(accuracy1, accuracy2)

    plot_data(Xtrain, Ttrain)
    plt.suptitle("Decision boundaries for Gaussian Naive Bayes")
    plt.show()
