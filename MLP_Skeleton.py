"""
Author: Ovunc Tuzel
"""

from __future__ import division
# from __future__ import print_function

import sys
import cPickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    def __init__(self, W, b):
        """
        - W is a nxm matrix where m is the number of hidden units, and n is the number of inputs.
        - b is a 1xm bias vector
        """

        self.W = W
        self.b = b

    def forward(self, x):
        """ Forward returns 1xm vector, z = Wx + b. """
        return np.add(np.matmul(x, self.W), self.b)

    def backward(
            self,
            grad_output,
            learning_rate=0.0,
            momentum=0.0,
            l2_penalty=0.0,
    ):
        pass


# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def forward(self, x):
        """ Receives 1xm vector, applies ReLU function to each element. """
        return np.maximum(x, np.zeros(x.shape[0]))

    def backward(
            self,
            grad_output,
            learning_rate=0.0,
            momentum=0.0,
            l2_penalty=0.0,
    ):
        pass


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def sigmoid(self, x):
        x = np.clip(x, -25, 25)
        return 1 / (1 + np.exp(-x))

    def xEntropy(self, a, y):
        # print "a", a
        # print "y", y
        # If else statement to avoid -infinities and division by zeros
        # a = np.clip(a, 10e-10, 1-10e-10)
        # print "\nPrediction ---- : ", a, "True ---- :", y
        xE = np.multiply(y, np.log(a)) + np.multiply(1 - y, np.log(1 - a))
        # print "LOSS:", -xE
        return -xE

        # xE = np.zeros((1, len(a[0])))
        # print a
        # for i in range(len(a[0])):
        #     if y[0, i] == 0:
        #         xE[0, i] = np.log(1 - a[0, i])
        #     elif y[0, i] == 1:
        #         xE[0, i] = np.log(a[0, i])
        #     else:
        #         print "ERROR: One hot encoding incorrect..."
        # print "LOSS:", xE
        # return xE

    def forward(self, x, y):
        """
        Follows a linear transform with a sigmoid activation.
        Then, A XEntropy cost function is applied w.r.t. one-hot encoded label y.
        """
        x = np.clip(x, -25, 25)
        z = 1 / (1 + np.exp(-x))
        return np.multiply(y, np.log(z)) + np.multiply(1 - y, np.log(1 - z))

    def backward(
            self,
            grad_output,
            learning_rate=0.0,
            momentum=0.0,
            l2_penalty=0.0
    ):
        pass
        # DEFINE backward function


# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):
    def __init__(self, input_dims, hidden_units, output_dims):
        self.learning_rate = 0.01
        self.dimensions = (input_dims, hidden_units, output_dims)
        self.W1 = np.random.randn(input_dims, hidden_units)
        self.b1 = np.random.randn(1, hidden_units)
        self.W2 = np.random.randn(hidden_units, output_dims)
        self.b2 = np.random.randn(1, output_dims)
        # self.h2 = np.zeros(1, output_dims)

    def forward(self, x):
        """ Get output from given input vector. """
        # print x
        z1 = LinearTransform(self.W1, self.b1).forward(x)
        # print z1
        a1 = ReLU().forward(z1)
        # print a1
        z2 = LinearTransform(self.W2, self.b2).forward(a1)
        # print "Before Sigmoid", z2
        y = SigmoidCrossEntropy().sigmoid(z2)
        # print "Output", y
        return y

    def get_gradients(self, y, a2, a1, z1, x):
        dLdW2 = np.matmul(a1.T, a2 - y)
        dLdb2 = a2 - y

        dLdW1 = np.matmul(np.matmul(x.T, a2-y), self.W2.T)
        dLdb1 = np.matmul(a2-y, self.W2.T)
        dLdW1 = np.asarray([dLdW1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdW1.T[j])) for j in range(len(dLdW1.T))]).T
        dLdb1 = np.asarray([dLdb1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdb1.T[j])) for j in range(len(dLdb1.T))]).T

        return dLdW1, dLdb1, dLdW2, dLdb2

    def get_gradients_Sigmoid(self, y, a2, a1, z1, x):
        dLdW2 = np.matmul(a1.T, a2 - y)
        dLdb2 = a2 - y

        dLdW1 = np.matmul(np.matmul(x.T, np.multiply(a2 - y, np.multiply(a2, (1-a2)))), self.W2.T)
        dLdb1 = np.matmul(a2 - y, self.W2.T)

        return dLdW1, dLdb1, dLdW2, dLdb2

    def backpropagate(self, L, y, a2, z2, a1, z1, x):
        """ Given a loss vector, update weights W1 and W2. """
        # Calc grads W2, b2
        # a1.T is (hidden# x 1) vector, a2 - y is (1 x output#) vector
        dLdW2 = np.matmul(a1.T, a2 - y)
        dLdb2 = a2 - y

        # xT is input# x 1
        # y-a2 is 1 x out#
        # W2T = out# x hid#

        dLdW1 = np.matmul(np.matmul(x.T, y - a2), self.W2.T)
        dLdb1 = np.matmul(y - a2, self.W2.T)
        #
        # # ReLu derivative - Zero out if z1[j] < 0 for w1jk
        dLdW1 = np.asarray([dLdW1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdW1.T[j])) for j in range(len(dLdW1.T))]).T
        dLdb1 = np.asarray([dLdb1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdb1.T[j])) for j in range(len(dLdb1.T))]).T

        self.W2 -= dLdW2 * self.learning_rate
        self.b2 -= dLdb2 * self.learning_rate
        self.W1 -= dLdW1 * self.learning_rate
        self.b1 -= dLdb1 * self.learning_rate

    def train_sgd(self, x, y, learning_rate=0.01):
        """
            z -> before activation
            a -> activated
        """
        z1 = LinearTransform(self.W1, self.b1).forward(x)
        a1 = ReLU().forward(z1)
        z2 = LinearTransform(self.W2, self.b2).forward(a1)
        a2 = SigmoidCrossEntropy().sigmoid(z2)
        L = SigmoidCrossEntropy().xEntropy(a2, y)
        self.backpropagate(L, y, a2, z2, a1, z1, x)

    def train_minibatch(self, x_batch, y_batch, learning_rate, momentum=[], l2_penalty=[]):
        dLdW1 = np.zeros((input_dims, hidden_units))
        dLdW2 = np.zeros((hidden_units, output_dims))
        dLdb1 = np.zeros((1, hidden_units))
        dLdb2 = np.zeros((1, output_dims))
        loss = 0
        for m in range(len(x_batch)):
            x = x_batch[m]
            y = y_batch[m]
            z1 = LinearTransform(self.W1, self.b1).forward(x)
            a1 = ReLU().forward(z1)
            # a1 = SigmoidCrossEntropy().sigmoid(z1)
            z2 = LinearTransform(self.W2, self.b2).forward(a1)
            a2 = SigmoidCrossEntropy().sigmoid(z2)
            # print "\n\n", "Pred", a2, "Input", x, "Output", y
            L = sum(SigmoidCrossEntropy().xEntropy(a2, y)[0])
            loss += L
            gW1, gb1, gW2, gb2 = self.get_gradients(y, a2, a1, z1, x)
            # gW1, gb1, gW2, gb2 = self.get_gradients_Sigmoid(y, a2, a1, z1, x)
            dLdW1 += gW1
            dLdb1 += gb1
            dLdW2 += gW2
            dLdb2 += gb2
        dLdW1 /= len(x_batch)
        dLdW2 /= len(x_batch)
        dLdb1 /= len(x_batch)
        dLdb2 /= len(x_batch)

        self.W2 -= dLdW2 * learning_rate
        self.b2 -= dLdb2 * learning_rate
        self.W1 -= dLdW1 * learning_rate
        self.b1 -= dLdb1 * learning_rate
        return loss
    # INSERT CODE for training the network

    def predict(self, x):
        return self.forward(x)

    def evaluate(self, x, y):
        pass
        # INSERT CODE for testing the network

        # ADD other operations and data entries in MLP if needed


# ===============================
class Visualizations(object):
    @staticmethod
    def plot_loss(losses):
        print losses
        plt.plot(losses)
        plt.show()

    @staticmethod
    def visualize_img(img, scale=5):
        """ Display 32x32x3 RGB image using OpenCV. """
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Display window", img)
        cv2.waitKey(0)

def unison_shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

# noinspection PyTypeChecker
class DataParser(object):
    def __init__(self, nn):
        self.mlp = nn

    def get_input(self, input_set, index=None):
        """ Get an input vector from a data set. Return a random input if the index is not specified. """
        if not index:
            index = np.random.randint(0, self.mlp.dimensions[0])
        input = input_set[index] / 255.0
        return input.reshape(1, len(input))

    def get_input_batch(self, input_set, batch_size=32):
        """ Get a minibatch from an input data set. """
        batch = input_set[0:batch_size]
        return batch.reshape(len(batch), 1, len(batch[0]))/255.0

    def get_input_batch_p(self, batch):
        """ Get a minibatch from an input data set. """
        return batch.reshape(len(batch), 1, len(batch[0]))/255.0

    def get_output(self, label_set, index=None):
        """
            Return a one-hot encoded output vector from a label set.
            Return a random sample if the index is not specified.
        """
        # print self.mlp.dimensions
        if not index:
            index = np.random.randint(0, self.mlp.dimensions[2])
        # print index
        label = np.zeros((1, self.mlp.dimensions[2]))
        # print label
        # print label_set
        label[0, label_set[index][0]] = 1
        return label

    def get_output_batch(self, label_set, batch_size=32):
        """ Get a minibatch from an output set. """
        batch = label_set[0:batch_size]
        labels = []
        for l in range(len(batch)):
            label = np.zeros((1, self.mlp.dimensions[2]))
            label[0, label_set[l][0]] = 1
            # print label
            # print baqtch[l]
            labels.append(label)
        return np.asarray(labels)

    def get_output_batch_p(self, batch):
        """ Get a minibatch from an output set. """
        labels = []
        for l in range(len(batch)):
            label = np.zeros((1, self.mlp.dimensions[2]))
            label[0, batch[l][0]] = 1
            # print label
            # print baqtch[l]
            labels.append(label)

        # print np.asarray(labels)q

        # label = np.zeros((1, self.mlp.dimensions[2]))
        # label[0, label_set[index][0]] = 1
        return np.asarray(labels)
# #
if __name__ == '__main__':
    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    np.random.seed(2)

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

    num_examples, input_dims = train_x.shape
    hidden_units = 50
    output_dims = 2

    num_epochs = 2000
    batch_size = 32
    learning_rate = 0.005

    mlp = MLP(input_dims, hidden_units, output_dims)

    losses = []

    print "ReLU {0} Batch, {1} LR, {2} Hidden Units".format(batch_size, learning_rate, hidden_units)
    ct = 0
    for i in range(num_epochs):
        p = np.random.choice(np.arange(0, num_examples), batch_size)
        x_batch = DataParser(mlp).get_input_batch_p(train_x[p])
        y_batch = DataParser(mlp).get_output_batch_p(train_y[p])
        l = mlp.train_minibatch(x_batch, y_batch, learning_rate)
        print ct, " LOSS: ", l/batch_size
        ct += 1
        losses.append(l/batch_size)
    Visualizations().plot_loss(losses)

    p = np.random.choice(np.arange(0, num_examples), 1000)
    x_batch = DataParser(mlp).get_input_batch_p(train_x[p])
    y_batch = DataParser(mlp).get_output_batch_p(train_y[p])
    ct =0
    for i in range(1000):
        out = mlp.predict(x_batch[i])[0]
        if out.argmax() == y_batch[i][0].argmax():
            ct+=1
    print "ACCURACY: ", ct/1000.0

    p = np.random.choice(np.arange(0, num_examples), 32)
    x_batch = DataParser(mlp).get_input_batch_p(train_x[p])
    for i in range(32):
        Visualizations.visualize_img(np.transpose(np.reshape(x_batch[i],(3,32,32)),(1,2,0)))
        out = mlp.predict(x_batch[i])[0]
        if (out.argmax()) == 0:
            print "AIRPLANE"
        else:
            print "SHIP"
