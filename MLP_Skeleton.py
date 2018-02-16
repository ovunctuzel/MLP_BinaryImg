"""
    Deep Learning HW #2
    Author: Ovunc Tuzel
    Email: tuzelo@oregonstate.edu
"""

from __future__ import division
import cPickle as Pickle
import numpy as np
import visualizations


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


class ReLU(object):
    @staticmethod
    def forward(x):
        """ Receives 1xm vector, applies ReLU function to each element. """
        return np.maximum(x, np.zeros(x.shape[0]))


# noinspection PyPep8Naming
class SigmoidCrossEntropy(object):
    @staticmethod
    def sigmoid(x):
        """ Applies sigmoid function to x. Clips the signal to avoid numerical errors. """
        x = np.clip(x, -25, 25)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def xEntropy(a, y):
        """ Returns the cross entropy error for binary classification. """
        xe = np.multiply(y, np.log(a)) + np.multiply(1 - y, np.log(1 - a))
        return -xe

    @staticmethod
    def forward(x, y):
        """
        Follows a linear transform with a sigmoid activation.
        Then, A XEntropy cost function is applied w.r.t. one-hot encoded label y.
        """
        x = np.clip(x, -25, 25)
        z = 1 / (1 + np.exp(-x))
        return np.multiply(y, np.log(z)) + np.multiply(1 - y, np.log(1 - z))


# noinspection PyTypeChecker
class MLP(object):
    def __init__(self, input_dims, hidden_units, output_dims):
        self.learning_rate = 0.01
        self.dimensions = (input_dims, hidden_units, output_dims)
        self.W1 = np.random.randn(input_dims, hidden_units)
        self.b1 = np.random.randn(1, hidden_units)
        self.W2 = np.random.randn(hidden_units, output_dims)
        self.b2 = np.random.randn(1, output_dims)
        #
        self.p_dLdW1 = np.zeros((input_dims, hidden_units))
        self.p_dLdb1 = np.zeros((1, hidden_units))
        self.p_dLdW2 = np.zeros((hidden_units, output_dims))
        self.p_dLdb2 = np.zeros((1, output_dims))

    def forward(self, x):
        """ Get output from given input vector. """
        z1 = LinearTransform(self.W1, self.b1).forward(x)
        a1 = ReLU().forward(z1)
        z2 = LinearTransform(self.W2, self.b2).forward(a1)
        y = SigmoidCrossEntropy().sigmoid(z2)
        return y

    def get_gradients(self, y, a2, a1, z1, x):
        dLdW2 = np.matmul(a1.T, a2 - y)
        dLdb2 = a2 - y

        dLdW1 = np.matmul(np.matmul(x.T, a2 - y), self.W2.T)
        dLdb1 = np.matmul(a2 - y, self.W2.T)
        dLdW1 = np.asarray([dLdW1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdW1.T[j])) for j in range(len(dLdW1.T))]).T
        dLdb1 = np.asarray([dLdb1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdb1.T[j])) for j in range(len(dLdb1.T))]).T

        return dLdW1, dLdb1, dLdW2, dLdb2

    def backprop_minibatch(self, x_batch, y_batch, learning_rate, momentum, l2_penalty=[]):
        input_dims, hidden_units, output_dims = self.dimensions
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
            z2 = LinearTransform(self.W2, self.b2).forward(a1)
            a2 = SigmoidCrossEntropy().sigmoid(z2)
            L = sum(SigmoidCrossEntropy().xEntropy(a2, y)[0])
            loss += L
            gW1, gb1, gW2, gb2 = self.get_gradients(y, a2, a1, z1, x)
            dLdW1 += gW1
            dLdb1 += gb1
            dLdW2 += gW2
            dLdb2 += gb2
        # Get the average gradient
        dLdW2 /= len(x_batch)
        dLdb2 /= len(x_batch)
        dLdW1 /= len(x_batch)
        dLdb1 /= len(x_batch)

        # Add previous gradients for momentum
        dLdW2 = momentum * self.p_dLdW2 + (1 - momentum) * dLdW2
        dLdb2 = momentum * self.p_dLdb2 + (1 - momentum) * dLdb2
        dLdW1 = momentum * self.p_dLdW1 + (1 - momentum) * dLdW1
        dLdb1 = momentum * self.p_dLdb1 + (1 - momentum) * dLdb1
        # Save previous gradients for momentum
        self.p_dLdW2 = dLdW2
        self.p_dLdb2 = dLdb2
        self.p_dLdW1 = dLdW1
        self.p_dLdb1 = dLdb1
        # Update weights
        self.W2 -= dLdW2 * learning_rate
        self.b2 -= dLdb2 * learning_rate
        self.W1 -= dLdW1 * learning_rate
        self.b1 -= dLdb1 * learning_rate
        return loss

    def train_minibatch(self, train_x, train_y, batch_size, num_epochs, learning_rate, momentum):
        losses = []
        for i in range(num_epochs):
            x_batch, y_batch = DataParser(self).get_input_output_batch(train_x, train_y, batch_size)
            l = self.backprop_minibatch(x_batch, y_batch, learning_rate, momentum)
            print len(losses), " LOSS: ", l / batch_size
            losses.append(l / batch_size)
        return losses

    # INSERT CODE for training the network

    def predict(self, x):
        return self.forward(x)

    def evaluate(self, test_x, test_y, examples=2000):
        x_batch, y_batch = DataParser(self).get_input_output_batch(test_x, test_y, examples)

        ct = 0
        for i in range(examples):
            out = self.predict(x_batch[i])[0]
            if out.argmax() == y_batch[i][0].argmax():
                ct += 1
        return ct / examples


class DataParser(object):
    def __init__(self, nn):
        self.mlp = nn

    def get_input(self, input_set, index=None):
        """ Get an input vector from a data set. Return a random input if the index is not specified. """
        if not index:
            index = np.random.randint(0, self.mlp.dimensions[0])
        input = input_set[index] / 255.0
        return input.reshape(1, len(input))

    def get_input_batch(self, batch):
        """ Get a minibatch from an input data set. """
        return batch.reshape(len(batch), 1, len(batch[0])) / 255.0

    def get_output(self, label_set, index=None):
        """
            Return a one-hot encoded output vector from a label set.
            Return a random sample if the index is not specified.
        """
        if not index:
            index = np.random.randint(0, self.mlp.dimensions[2])
        label = np.zeros((1, self.mlp.dimensions[2]))
        label[0, label_set[index][0]] = 1
        return label

    def get_output_batch(self, batch):
        """ Get a minibatch from an output set. """
        labels = []
        for l in range(len(batch)):
            label = np.zeros((1, self.mlp.dimensions[2]))
            label[0, batch[l][0]] = 1
            labels.append(label)
        return np.asarray(labels)

    def get_input_output_batch(self, train_x, train_y, batch_size):
        """ Get a tuple of minibatches from an output set. """
        p = np.random.choice(np.arange(0, train_x.shape[0]), batch_size)
        x_batch = self.get_input_batch(train_x[p])
        y_batch = self.get_output_batch(train_y[p])
        return x_batch, y_batch


def train():
    """ Train a mlp with minibatch gradient descent. """
    print "ReLU {0} Batch, {1} LR, {2} Hidden Units".format(batch_size, learning_rate, hidden_units)
    mlp = MLP(input_dims, hidden_units, output_dims)
    losses = mlp.train_minibatch(train_x, train_y, batch_size, num_epochs, learning_rate, momentum)

    # Visualize Learning Curve
    visualizations.plot_loss(losses, save=True, specs=(momentum, batch_size, learning_rate, hidden_units))
    # Save Network
    Pickle.dump(mlp,
                open("NNs/{0}Mom_{1}Batch_{2}LR_{3}_Hid.p".format(momentum, batch_size, learning_rate, hidden_units),
                     "wb"))


def test():
    """ Evaluate an mlp. """
    # Load Network
    print "MLP_{0}Batch_{1}LR_{2}_Hid.p".format(batch_size, learning_rate, hidden_units)
    mlp = Pickle.load(
        open("NNs/{0}Mom_{1}Batch_{2}LR_{3}_Hid.p".format(momentum, batch_size, learning_rate, hidden_units), "rb"))
    success = mlp.evaluate(test_x, test_y, 2000)
    print "Success Rate: ", success
    filename = "Results/results.txt"
    file = open(filename, "a")
    file.write("{},{},{},{},{}\n".format(momentum, batch_size, learning_rate, hidden_units, success))


if __name__ == '__main__':
    # PARAMETERS #######################################
    hidden_units = 3
    output_dims = 2
    num_epochs = 5000
    batch_size = 32
    learning_rate = 0.01
    momentum = 0.2
    ####################################################


    data = Pickle.load(open('cifar_2class_py2.p', 'rb'))

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
    num_examples, input_dims = train_x.shape

    train()
    test()
