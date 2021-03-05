import numpy as np
import numpy.random as rnd
import pickle 
import matplotlib.pyplot as plt
import sklearn.neural_network as nn
import sklearn.utils as utls 


def sigmoid(z):
    return 1/(1 + np.exp(-z)) 

def print_accuracy(accuracy1, accuracy2):
    print("sklearn accuracy: {}".format(accuracy1))
    print("model accuracy: {}".format(accuracy2))
    print("accuracy difference: {}".format(abs(accuracy1 - accuracy2)))

def print_CE(CE1, CE2):
    print("sklearn CE: {}".format(CE1))
    print("model CE: {}".format(CE2))
    print("CE difference: {}".format(abs(CE1 - CE2)))

def reduce_data(X, T, val1, val2):
    new_X = X[(T == val1) | (T == val2)]
    new_T = T[(T == val1) | (T == val2)]
    new_T = np.where(new_T == val1, 1, 0)

    return new_X, new_T

def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z)) 

def fwpropagate(X, W, b):

    h1 = tanh(X @ W[0] + b[0])
    h2 = tanh(h1 @ W[1] + b[1])
    O = sigmoid(h2 @ W[2] + b[2])
    
    return O 

def accuracyNN_Q5(X, T, W, b):

    # Propagate for 2 hidden layers 
    Y = fwpropagate(X, W, b)

    pred = np.where(Y >= 0.5, 1, 0)
    accuracy = np.mean(pred.T == T)

    return accuracy

def ceNN_Q5(X, T, W, b):

    # Propagate for 2 hidden layers 
    Y = fwpropagate(X, W, b)

    # Cross entropy
    CE = (-T * np.log(Y).T) - ((1 - T) * np.log(1 - Y).T)
    return np.mean(CE)

def evaluateNN(clf, X, T):

    # Weights & Bias
    W = clf.coefs_
    b = clf.intercepts_

    # Calculate accuracies
    accuracy1 = clf.score(X, T)
    accuracy2 = accuracyNN_Q5(X, T, W, b)

    # Calculate CE
    CE1_logprob = clf.predict_log_proba(X)[np.arange(len(T)), T]
    CE1 = np.mean(-1 * CE1_logprob)
    CE2 = ceNN_Q5(X, T, W, b)

    return accuracy1, accuracy2, CE1, CE2


def script_NN():
    np.random.seed(0)
    clf = nn.MLPClassifier(hidden_layer_sizes=(100, 100), batch_size=100, max_iter=100, activation="tanh", solver="sgd", learning_rate_init=0.01, tol=10**(-6))

    with open("./data/mnist.pickle","rb") as file:
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(file)
        
    # Training and Test sets for 5,6
    X_train, T_train = reduce_data(Xtrain, Ttrain, 5, 6)
    X_test, T_test = reduce_data(Xtest, Ttest, 5, 6)

    clf.fit(X_train, T_train)
    NN_W, NN_b = clf.coefs_, clf.intercepts_

    accuracy1, accuracy2, CE1, CE2 = evaluateNN(clf, X_test, T_test)

    print_accuracy(accuracy1, accuracy2)
    print_CE(CE1, CE2)


def backpropagate(A_s, Ws, bs, err):
    # Activation functions from forward propogation
    X, H, G, O = A_s[0], A_s[1], A_s[2], A_s[3]
    # Prior weights
    W, V, U = Ws[0], Ws[1], Ws[2]

    # Output layer
    dU = G.T @ err 
    du0 = np.sum(err, axis=0)

    dG = err @ U.T 
    dH_ = (1 - G**2) * dG

    # Hidden layer 2
    dV = H.T @ dH_
    dv0 = np.sum(dH_, axis=0)

    dH = dH_ @ V.T
    dX_ = (1 - H**2) * dH 

    # Hidden layer 1
    dW = X.T @ dX_
    dw0 = np.sum(dX_, axis=0)

    weights = np.array([dW, dV, dU])
    bias = np.array([dw0, dv0, du0])
    
    return weights, bias

def init_weights(mean, var, Wsize, bsize):
    W = np.random.normal(mean, var, np.shape(Wsize[0]))
    V = np.random.normal(mean, var, np.shape(Wsize[1]))
    U = np.random.normal(mean, var, np.shape(Wsize[2]))

    w0 = np.zeros(np.shape(bsize[0]))
    v0 = np.zeros(np.shape(bsize[1]))
    u0 = np.zeros(np.shape(bsize[2]))
    
    Ws = np.array([W, V, U])
    bs = np.array([w0, v0, u0])

    return Ws, bs

def batch_GD(iter, lrate, Wsize, bsize, Xtrain, Ttrain, Xtest, Ttest):
    np.random.seed(0)
    
    Ws, bs = init_weights(0, 1, Wsize, bsize)
    N = len(Ttrain)

    # Stepping
    for i in range(iter):
        accuracy = accuracyNN_Q5(Xtest, Ttest, Ws, bs)
        print("iteration #: {}".format(i))
        print("accuracy: {}".format(accuracy))

        # Forward propagation
        H = tanh(Xtrain @ Ws[0] + bs[0])
        G = tanh(H @ Ws[1] + bs[1])
        O = sigmoid(G @ Ws[2] + bs[2])
        As = [Xtrain, H, G, O]

        # Error signal of output
        err = O.T - Ttrain

        # Backward propagation
        grad_weights, grad_bias = backpropagate(As, Ws, bs, err.T)
        
        # Update weights and bias
        Ws -= (lrate/N) * grad_weights
        bs -= (lrate/N) * grad_bias

    accuracy = accuracyNN_Q5(Xtest, Ttest, Ws, bs)
    CE = ceNN_Q5(Xtest, Ttest, Ws, bs)
    return accuracy, CE

def script_batch_GD():
    np.random.seed(0)
    clf = nn.MLPClassifier(hidden_layer_sizes=(100, 100), batch_size=100, max_iter=100, activation="tanh", solver="sgd", learning_rate_init=0.01, tol=10**(-6))

    with open("./data/mnist.pickle","rb") as file:
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(file)
        
    # Training and Test sets for 5,6
    X_train, T_train = reduce_data(Xtrain, Ttrain, 5, 6)
    X_test, T_test = reduce_data(Xtest, Ttest, 5, 6)

    clf.fit(X_train, T_train)
    NN_W, NN_b = clf.coefs_, clf.intercepts_

    accuracy, CE = batch_GD(10, 0.1, NN_W, NN_b, X_train, T_train, X_test, T_test)
    print("accuracy: {}".format(accuracy))
    print("CE: {}".format(CE))


def sgd(epoch, minibatch_size, lrate, Wsize, bsize, Xtrain, Ttrain, Xtest, Ttest):
    np.random.seed(0)
    Ws, bs = init_weights(0, 1, Wsize, bsize)

    # Epoch begins
    for i in range(epoch):
        accuracy = accuracyNN_Q5(Xtest, Ttest, Ws, bs)
        print("epoch #: {}".format(i))
        print("accuracy: {}".format(accuracy))

        # Shuffle Data
        Xtrain, Ttrain = utls.shuffle(Xtrain, Ttrain)
        
        # Stepping
        for j in range(0, np.shape(Xtrain)[0], minibatch_size):
            # Create mini-batches
            Xtrain_mini = Xtrain[j:j+minibatch_size]
            Ttrain_mini = Ttrain[j:j+minibatch_size]

            # Forward propagation
            H = tanh(Xtrain_mini @ Ws[0] + bs[0])
            G = tanh(H @ Ws[1] + bs[1])
            O = sigmoid(G @ Ws[2] + bs[2])
            As = [Xtrain_mini, H, G, O]

            # Error signal of output
            err = O.T - Ttrain_mini

            # Backward propagation
            grad_weights, grad_bias = backpropagate(As, Ws, bs, err.T)
            
            # Update weights and bias
            Ws -= (lrate/minibatch_size) * grad_weights
            bs -= (lrate/minibatch_size) * grad_bias

    accuracy = accuracyNN_Q5(Xtest, Ttest, Ws, bs)
    CE = ceNN_Q5(Xtest, Ttest, Ws, bs)
    return accuracy, CE


def script_sgd():
    np.random.seed(0)
    clf = nn.MLPClassifier(hidden_layer_sizes=(100, 100), batch_size=100, max_iter=100, activation="tanh", solver="sgd", learning_rate_init=0.01, tol=10**(-6))

    with open("./data/mnist.pickle","rb") as file:
        Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(file)
        
    # Training and Test sets for 5,6
    X_train, T_train = reduce_data(Xtrain, Ttrain, 5, 6)
    X_test, T_test = reduce_data(Xtest, Ttest, 5, 6)

    clf.fit(X_train, T_train)
    NN_W, NN_b = clf.coefs_, clf.intercepts_

    accuracy, CE = sgd(10, 10, 0.1, NN_W, NN_b, X_train, T_train, X_test, T_test) 
    print("accuracy: {}".format(accuracy))
    print("CE: {}".format(CE))