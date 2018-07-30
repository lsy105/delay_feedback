import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
# import encoding
import PSDD
import reservoir


class DFRSpiking:

    b = np.zeros((18, 4096))
    D1 = np.zeros((18, 4096, 3))
    TS = np.zeros((18, 4096, 3))
    IPSC = np.zeros((18, 4096, 76))

    def import_images(self):
        for i in range(18):
            im = plt.imread(str(i + 1) + '.jpg')
            im2 = self.sp_noise(im, 0.9)
            im2 = np.reshape(im2, -1)
            self.b[i] = im2

    def sp_noise(self, image, prob):
        sh = image.shape
        output = np.zeros(sh, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def test_noise(self):
        im = plt.imread('1.jpg')
        ret = self.sp_noise(im, 0.9)
        img = Image.fromarray(ret, 'L')
        img.save('noised_1.jpg')
        img.show()

    def encoding(self, value):
        return np.array((value/9, value*2/9, value/9))

    def apply_encoding(self):
        for i in range(18):
            for j in range(4096):
                self.D1[i, j, :] = self.encoding(self.b[i, j])

    # def apply_encoding(self, array):
    #     ret = np.zeros((len(array), 3))
    #     for i in range(len(array)):
    #         ret[i, :] = self.encoding(array[i])
    #     return ret

    def test_encoding(self):
        im = plt.imread('1.jpg')
        im2 = self.sp_noise(im, 0.9)
        im2 = np.reshape(im2, -1)
        # im3 = self.apply_encoding(im2)
        # return im3

    def assign_ts(self):
        for i in range(18):
            for j in range(4096):
                self.TS[i, j, 0] = self.D1[i, j, 0]
                for t in range(1, 3):
                    self.TS[i, j, t] = self.TS[i, j, t-1] + self.D1[i, j, t]

    def assign_IPSC(self):
        for i in range(18):
            for p in range(4096):
                self.IPSC[i, p, :] = PSDD.PSDD(self.TS[i, p, :])
        return self.IPSC


def main():
    d1 = DFRSpiking()
    d1.import_images()

    d1.apply_encoding()
    # print(d1.D1)
    d1.assign_ts()
    ipsc = d1.assign_IPSC()  # pass this ipsc to next package reservior.m
    # Design: make reservior and NN train as two classes, just pass matrix
    r1 = reservoir.Reservoir(ipsc)
    XX0delay20test, Rate0Delay20test = r1.assign_values()


if __name__ == '__main__':
    main()