import numpy as np
import math


class NNtrain:
    def __init__(self):
        infile_train = 'XX0delay20_train.npy'
        infile_test = 'XX0delay20_test.npy'

        # XX stores time of spikes, 18 is num of images, 4096 is num of pixels, and 13 is spikes of that pixel
        # reshape to 53248, which is 4096 * 13
        self.Datatrain = np.zeros((18, 53248))  # 53248 = 4096 * 13
        self.XX1 = np.load(infile_train)
        self.Datatest = np.zeros((18, 53248))
        self.XXtest = np.load(infile_test)
        self.Aeta = 1e-3
        self.MaxE = 1e-2
        self.alpha = 0.7

        # Represent classes, but for the 18 images dataset, only 3 classes, so only 00, 01, 10 are used.
        self.DesireOutput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.randDesireOutput = np.zeros(2)
        self.P, self.NumofInputNeuron = self.Datatrain.shape
        self.HiddenLayer1 = 20
        self.HiddenLayer2 = 10
        self.NumofOutputNeuron = 2

        # v1, v2, vn are the fully connected layers' weights, Dv1, Dv2, Dvn are differences used to update weights.
        self.v1 = 0.01 * np.random.randn(self.NumofInputNeuron + 1, self.HiddenLayer1)
        self.Dv1 = np.zeros((self.NumofInputNeuron + 1, self.HiddenLayer1))
        self.v2 = 0.01 * np.random.randn(self.HiddenLayer1 + 1, self.HiddenLayer2)
        self.Dv2 = np.zeros((self.HiddenLayer1 + 1, self.HiddenLayer2))
        self.vn = 0.01 * np.random.randn(self.HiddenLayer2 + 1, self.NumofOutputNeuron)
        self.Dvn = np.zeros((self.HiddenLayer2 + 1, self.NumofOutputNeuron))
        self.counter = 0
        self.E = 0
        self.E1 = 0
        self.F = True
        self.p = 0

        self.ee = np.empty([0])
        self.ee1 = np.empty([0])

    # This is not normalization, it's standardization.
    @staticmethod
    def standardize(vec):
        avg = np.mean(vec)
        std = np.std(vec)
        return (vec - avg) / std

    # reshape XX1 and XXtest to fit Datatrain and Datatest
    def assign_values(self):
        self.Datatrain = self.XX1.reshape((18, -1))
        self.Datatest = self.Datatest.reshape((18, -1))

        for i in range(18):
            self.Datatrain[i, :] = self.standardize(self.Datatrain[i, :])
            self.Datatest[i, :] = self.standardize(self.Datatrain[i, :])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def train_nn(self):
        while self.F:
            while self.p < self.P:
                if 0 <= self.p < 7:  # [0, 6]
                    self.randDesireOutput = self.DesireOutput[0, :]
                elif 6 < self.p < 13:  # [7, 12]
                    self.randDesireOutput = self.DesireOutput[1, :]
                elif 12 < self.p < 19:  # [13, 18]
                    self.randDesireOutput = self.DesireOutput[2, :]
                elif 18 < self.p < 25:  # [19, 24]
                    self.randDesireOutput = self.DesireOutput[3, :]

                x = self.Datatrain[self.p, :]
                y1 = NNtrain.sigmoid(np.dot(np.append(x, 1), self.v1))
                y2 = NNtrain.sigmoid(np.dot(np.append(y1, 1), self.v2))
                z = NNtrain.sigmoid(np.dot(np.append(y2, 1), self.vn))
                deltaz = self.randDesireOutput - z
                self.E += np.sum(np.power(deltaz, 2))  # np.power is elementwise power.

                deltaZ = np.dot(np.dot(z, 1-z), deltaz)
                Y2 = np.append(y2, 1)  # np.multiply is elementwise multiply.
                deltaY2 = np.multiply(np.multiply(Y2, (1-Y2)), np.dot(deltaZ, self.vn.T))
                deltaY2 = np.delete(deltaY2, self.HiddenLayer2)

                Y1 = np.append(y1, 1)
                deltaY1 = np.multiply(np.multiply(Y1, (1-Y1)), np.dot(deltaY2, self.v2.T))
                deltaY1 = np.delete(deltaY1, self.HiddenLayer1)

                self.Dvn = np.outer(np.multiply(self.Aeta, Y2.T), deltaZ) + np.multiply(self.alpha, self.Dvn)
                self.vn += self.Dvn

                self.Dv2 = np.outer(np.multiply(self.Aeta, Y1.T), deltaY2) + np.multiply(self.alpha, self.Dv2)
                self.v2 += self.Dv2

                X = np.append(x, 1)
                self.Dv1 = np.outer(np.multiply(self.Aeta, X.T), deltaY1) + np.multiply(self.alpha, self.Dv1)
                self.v1 += self.Dv1

                x11 = self.Datatest[self.p, :]
                y11 = NNtrain.sigmoid(np.dot(np.append(x11, 1), self.v1))
                y22 = NNtrain.sigmoid(np.dot(np.append(y11, 1), self.v2))
                z11 = NNtrain.sigmoid(np.dot(np.append(y22, 1), self.vn))
                deltaz1 = self.randDesireOutput - z11
                self.E1 += np.sum(np.power(deltaz1, 2))
                self.p += 1

            for tt in range(18):
                x1 = self.Datatest[tt, :]
                y11 = NNtrain.sigmoid(np.dot(np.append(x1, 1), self.v1))
                y22 = self.__class__.sigmoid(np.dot(np.append(y11, 1), self.v2))
                z1 = self.__class__.sigmoid(np.dot(np.append(y22, 1), self.vn))

            e1 = math.sqrt(self.E / (self.NumofOutputNeuron * self.P))
            self.ee = np.append(self.ee, e1)
            e11 = math.sqrt(self.E1 / (self.NumofOutputNeuron * self.P))
            self.ee1 = np.append(self.ee1, e11)

            print('ee: ', e1)
            print('ee1: ', e11)

            if e1 <= self.MaxE:
                self.F = False
            else:
                self.E = 0
                self.E1 = 0
                self.p = 0
                self.counter += 1


def main():
    nn = NNtrain()
    nn.assign_values()
    nn.train_nn()


if __name__ == '__main__':
    main()