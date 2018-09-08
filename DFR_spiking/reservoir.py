import numpy as np
import LIF1
import PSDD1
import matplotlib.pyplot as plt


class Reservoir(object):
    # Build the reservoir
    def __init__(self):
        infile = 'IPSC9delay30_test.npy'
        self.Data = np.load(infile)
        # self.gain = 0.8

        # XX stores time of spikes, 18 is num of images, 4096 is num of pixels, and 13 is spikes of that pixel
        self.XX = np.zeros((18, 4096, 13))
        self.X1 = np.zeros((18, 4096, 13))
        self.X2 = np.zeros((18, 4096, 13))
        self.X3 = np.zeros((18, 4096, 13))
        self.X4 = np.zeros((18, 4096, 13))
        self.Inew = np.zeros((18, 4096, 76))

    def assign_values(self):
        for i in range(4096):
            for j in range(6):
                if j == 0:
                    TT = LIF1.LIF1(16 * self.Data[0, i, :])
                    self.XX[0, i, 0:len(TT)] = TT
                    self.X1[0, i, 0:len(TT)] = TT + 20
                    self.X2[0, i, 0:len(TT)] = TT + 40
                    self.X3[0, i, 0:len(TT)] = TT + 60
                    self.X4[0, i, 0:len(TT)] = TT + 80
                    self.Inew[0, i, :] = PSDD1.psdd1(TT)
                else:
                    TT = LIF1.LIF1(16 * self.Data[j, i, :] + 0.8 * self.Inew[j-1, i, :])
                    self.XX[j, i, 0:len(TT)] = TT
                    self.X1[j, i, 0:len(TT)] = TT + 20
                    self.X2[j, i, 0:len(TT)] = TT + 40
                    self.X3[j, i, 0:len(TT)] = TT + 60
                    self.X4[j, i, 0:len(TT)] = TT + 80
                    self.Inew[j, i, :] = PSDD1.psdd1(self.X4[j, i, 0: len(TT)])

            for j in range(6, 12):
                if j == 6:
                    TT = LIF1.LIF1(16* self.Data[6, i, :])
                    self.XX[6, i, 0:len(TT)] = TT
                    self.X1[6, i, 0:len(TT)] = TT + 20
                    self.X2[6, i, 0:len(TT)] = TT + 40
                    self.X3[6, i, 0:len(TT)] = TT + 60
                    self.X4[6, i, 0:len(TT)] = TT + 80
                    self.Inew[6, i, :] = PSDD1.psdd1(TT)
                else:
                    TT = LIF1.LIF1(16 * self.Data[j, i, :] + 0.8 * self.Inew[j - 1, i, :])
                    self.XX[j, i, 0:len(TT)] = TT
                    self.X1[j, i, 0:len(TT)] = TT + 20
                    self.X2[j, i, 0:len(TT)] = TT + 40
                    self.X3[j, i, 0:len(TT)] = TT + 60
                    self.X4[j, i, 0:len(TT)] = TT + 80
                    self.Inew[j, i, :] = PSDD1.psdd1(self.X4[j, i, 0: len(TT)])

            for j in range(12, 18):
                if j == 12:
                    TT = LIF1.LIF1(16 * self.Data[12, i, :])
                    self.XX[12, i, 0:len(TT)] = TT
                    self.X1[12, i, 0:len(TT)] = TT + 20
                    self.X2[12, i, 0:len(TT)] = TT + 40
                    self.X3[12, i, 0:len(TT)] = TT + 60
                    self.X4[12, i, 0:len(TT)] = TT + 80
                    self.Inew[12, i, :] = PSDD1.psdd1(TT)
                else:
                    TT = LIF1.LIF1(16 * self.Data[j, i, :] + 0.8 * self.Inew[j - 1, i, :])
                    self.XX[j, i, 0:len(TT)] = TT
                    self.X1[j, i, 0:len(TT)] = TT + 20
                    self.X2[j, i, 0:len(TT)] = TT + 40
                    self.X3[j, i, 0:len(TT)] = TT + 60
                    self.X4[j, i, 0:len(TT)] = TT + 80
                    self.Inew[j, i, :] = PSDD1.psdd1(self.X4[j, i, 0: len(TT)])

        outfile = 'XX0delay20_test.npy'
        np.save(outfile, self.XX)
        return self.XX

    def test_data(self):
        x = 0  # should be between [0, 17]
        y = 30  # should be between [0, 4095]
        num = 5
        for i in range(num):
            plt.subplot(num, 1, i + 1)
            plt.plot(self.Data[x, y + i, :], c=np.random.rand(3,))
            plt.grid()
        plt.show()


def main():
    res = Reservoir()
    # res.test_data()
    xx = res.assign_values()
    print(xx)


if __name__ == '__main__':
    main()
