# import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class FDROriginal:
    # Parameters
    n = 10
    nv = n - 1
    k1 = 4 * 10
    k2 = 4 * k1
    k3 = 4 * k1
    k = k1 + k2 + k3
    gain = 0.8
    scale = 1
    m = np.array(nv)
    j = np.zeros(k*nv)

    input = np.array(k)
    input1 = np.array(k1)
    input2 = np.array(k2)
    input3 = np.array(k3)
    target = np.array(k)
    target1 = np.array(k2)
    target2 = np.array(k3)
    # Reservoir parameters
    x = np.zeros(n)  # states in NL
    x_next = np.zeros(n)
    X = np.zeros((k2, n))
    x_all = np.zeros((k2 * nv, n))  # states in all nodes

    reg = 1e-8  # regularization coefficient
    w_out = np.zeros(n)

    temp = np.zeros(k2)
    temp2 = np.zeros(k2)

    output_train = np.zeros(k2)
    X_all_test = np.zeros((k3 * nv, n))
    j_test = np.zeros(k3 * nv)

    X_test = np.zeros((k3, n))
    output_test = np.zeros(k3)

    # Define input & actual output data
    # NARMA10 task- nonlinear auto-regressive moving average
    def define_input_output(self):
        # seed = 100
        self.input, self.target = self.narma10_create(self)
        self.input1 = self.input[0: self.k1]
        self.input2 = self.input[self.k1: self.k1 + self.k2]
        self.input3 = self.input[self.k1 + self.k2:]

        self.target1 = self.target[self.k1: self.k1 + self.k2]
        self.target2 = self.target[self.k1 + self.k2:]

    @staticmethod
    def narma10_create(self):
        #     if second input is present, use it as the random seed
        #     this allows the use of the same detaset between trails

        input = 0.5 * np.random.uniform(low=0.0, high=1.0, size=self.k)
        outpout = np.zeros(self.k)

        for i in range(9, self.k-1):
            outpout[i+1] = 0.3 * outpout[i]\
                           + 0.05 * outpout[i] * sum(outpout[i-9:i]) + 1.5 * input[i] * input[i-9] + 0.1

        return input, outpout

    # Defining mask and masking the input
    def mask(self):
        # j is the masked input
        self.m = np.random.uniform(low=0.0, high=1.0, size=self.nv)
        it = np.nditer(self.input, flags=['f_index'])
        while not it.finished:
            masked_elem = np.dot(it[0], self.m)
            for index, value in enumerate(masked_elem):
                self.j[it.index * self.nv + index] = value
            it.iternext()

    # Initialize reservoir
    def init_reservoir(self):
        for i in range(0, self.k1 * self.nv):
            self.x_next[0] = np.tanh(self.scale * self.j[i] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]
            self.x = self.x_next

    # Training data through the reservoir and store the node states.
    def train_reservoir(self):
        for i in range(0, self.k2 * self.nv):
            t = self.k1 * self.nv + i
            self.x_next[0] = np.tanh(self.scale * self.j[t] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]  # in matlab, ] is inclusive!
            self.x = self.x_next
            self.x_all[i, :] = self.x

    # Consider the data just once everytime it loops around?
    def sample_feature(self):
        self.temp = np.arange(0, self.k2, 1)
        self.temp2 = self.nv * self.temp
        self.X[self.temp, :] = self.x_all[self.temp2, :]

    # Train the output weights
    def train_output_weights(self):
        # Here we use regularized least squares.
        numerator = np.dot(self.target1, self.X)
        denominator1 = np.dot(np.transpose(self.X), self.X)
        denominator2 = self.reg * np.identity(self.n)
        denominator = np.add(denominator1, denominator2)
        self.w_out, d1, d3, d4 = np.linalg.lstsq(denominator, numerator, rcond=None)
        return self.w_out, d1, d3, d4

    # Compute training error
    def training_error(self):
        self.output_train = np.dot(self.w_out, np.transpose(self.X))
        errorLen = self.k2
        mse = (LA.norm(self.target1 - self.output_train, 2)**2) / errorLen
        nmse = (LA.norm(self.target1 - self.output_train)/LA.norm(self.target1))**2
        return mse, nmse

    # Testing data through reservoir
    def test(self):
        self.x = np.zeros(self.n)
        self.x_next = np.zeros(self.n)
        self.j_test = self.j[self.nv*(self.k1+self.k2):len(self.j)]

        # Reservoir initialization
        for i in range(0, self.k1 * self.nv):
            self.x_next[0] = np.tanh(self.scale * self.j[i] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]
            self.x = self.x_next
            self.X_all_test[i, :] = self.x

        # Run data through the reservoir and store the node states.
        for i in range(0, self.k3 * self.nv):
            self.x_next[0] = np.tanh(self.scale * self.j_test[i] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]
            self.x = self.x_next
            self.X_all_test[i, :] = self.x

        # Consider the data just once everytime it loops around?
        self.temp = np.arange(0, self.k3, 1)
        self.temp2 = self.nv * self.temp
        self.X_test[self.temp, :] = self.X_all_test[self.temp2, :]
        error_len = self.k3
        self.output_test = np.dot(self.w_out, np.transpose(self.X_test))
        mse_test = (LA.norm(self.target2 - self.output_test, 2) ** 2) / error_len
        nmse_test = (LA.norm(self.target2 - self.output_test) / LA.norm(self.target2)) ** 2
        return mse_test, nmse_test

    def draw_charts(self):
        x1 = np.arange(0, 100, 1)
        x2 = np.arange(0, self.k2, 1)
        x41 = np.arange(0, self.k1, 1)
        x42 = np.arange(self.k1, self.k1 + self.k2, 1)
        x43 = np.arange(self.k1 + self.k2, self.k, 1)

        plt.figure(1)
        plt.plot(x1, self.j[self.k1 * self.nv: self.k1 * self.nv + 100], marker='x', c=np.random.rand(3,))
        plt.grid()
        plt.title('Input after Masking')

        plt.figure(2)
        plt.plot(x2, self.output_train, marker='o', c=np.random.rand(3,))
        plt.plot(x2, self.target1, marker='x', c=np.random.rand(3,))
        plt.xlabel('o: Reservoir, x: Actual')
        plt.grid()
        plt.title('Train: Reservoir Output vs Actual Output for NARMA-input')

        plt.figure(3)
        plt.plot(x2, self.output_test, marker='o', c=np.random.rand(3,))
        plt.plot(x2, self.target2, marker='x', c=np.random.rand(3,))
        plt.xlabel('o: Reservoir, x: Actual')
        plt.grid()
        plt.title('Test: Reservoir Output vs Actual Output for NARMA-input')

        plt.figure(4)
        plt.plot(x41, self.input1, c=np.random.rand(3,))
        plt.plot(x42, self.input2, c=np.random.rand(3,))
        plt.plot(x43, self.input3, c=np.random.rand(3,))
        plt.xlabel('Left: Initial, Middle: Training, Right: Testing')
        plt.grid()
        plt.title('Input Sequence')

        plt.figure(5)
        plt.plot(x41, self.target[0: self.k1], c=np.random.rand(3, ))
        plt.plot(x42, self.target1, c=np.random.rand(3, ))
        plt.plot(x43, self.target2, c=np.random.rand(3, ))
        plt.xlabel('Left: Initial, Middle: Training, Right: Testing')
        plt.grid()
        plt.title('Target Sequence')
        plt.show()


def test_divide():
    a = np.arange(1, 10, 1).reshape((3, 3))
    b = np.arange(5, 8, 1)
    c = np.dot(a, b)
    # d = np.divide(c, a)
    d1, d3, d4, d5 = np.linalg.lstsq(a, c)
    print(c)
    print(d1, d3, d4, d5)


def main():
    f1 = FDROriginal()
    f1.define_input_output()
    f1.mask()
    f1.init_reservoir()
    f1.train_reservoir()
    f1.sample_feature()
    f1.train_output_weights()
    mse, nmse = f1.training_error()
    print('mse: ', mse, ', nmse: ',  nmse)
    mse_test, nmse_test = f1.test()
    print('mse_test: ', mse_test, ', nmse_test: ', nmse_test)
    f1.draw_charts()
    # test_divide()


if __name__ == "__main__":
    main()

