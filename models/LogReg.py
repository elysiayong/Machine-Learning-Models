import numpy as np
import numpy.random as rnd
import pickle 
import matplotlib.pyplot as plt
import sklearn.linear_model as lin


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
    print("model accuracy: {}".format(accuracy2))
    print("accuracy difference: {}".format(abs(accuracy1 - accuracy2)))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)

def accuracyLR(clf, X, T):
    W = clf.coef_ 
    b = clf.intercept_ 
    
    Z = (X @ W.T) + b
    Y = softmax(Z)
    accuracy = np.mean(np.argmax(Y, axis=len(np.shape(Y))-1) == T)
    
    return accuracy


def script_logreg():
    with open("./data/data2.pickle","rb") as file:
        dataTrain, dataTest = pickle.load(file)
    Xtrain, Ttrain = dataTrain
    Xtest, Ttest = dataTest

    clf = lin.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf.fit(Xtrain, Ttrain)

    accuracy1 = clf.score(Xtest, Ttest)
    accuracy2 = accuracyLR(clf, Xtest, Ttest)
    print_accuracy(accuracy1, accuracy2)

    plot_data(Xtrain, Ttrain)
    plt.suptitle("Decision boundaries for logistic regression")
    plt.show()

