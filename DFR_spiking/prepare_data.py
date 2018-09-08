import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import PSDD


class DFRSpiking:

    def __init__(self):
        self.b = np.zeros((18, 4096))
        self.D1 = np.zeros((18, 4096, 3))  # intervals between spikes
        self.TS = np.zeros((18, 4096, 3))  # time of spikes, real timestamps
        self.IPSC = np.zeros((18, 4096, 76))  # stores analog data

    def import_images(self):
        for i in range(18):
            im = plt.imread(str(i + 1) + '.jpg')
            im = self.sp_noise(im, 0.5)  # add noise to test images, no noise to train images
            im2 = np.reshape(im, -1)
            self.b[i] = im2

    def sp_noise(self, image, prob):
        sh = image.shape
        output = np.zeros(sh, np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rand1 = random.random()
                if rand1 < prob:
                    rand2 = random.random()
                    if rand2 < 0.5:
                        output[i][j] = 0
                    else:
                        output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def test_noise(self):
        for i in range(18):
            im = plt.imread(str(i + 1) + '.jpg')
            im = self.sp_noise(im, 0.9)
            img = Image.fromarray(im, 'L')
            img.save('noised_' + str(i + 1) + '.jpg')

    def encoding(self, value):
        # convert pixels to spikes
        return np.array((value/9, value*2/9, value/9))

    def apply_encoding(self):
        for i in range(18):
            for j in range(4096):
                self.D1[i, j, :] = self.encoding(self.b[i, j])

    def test_encoding(self):
        im = plt.imread('1.jpg')
        im2 = self.sp_noise(im, 0.9)
        im2 = np.reshape(im2, -1)

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

        outfile = 'IPSC9delay30_test.npy'
        np.save(outfile, self.IPSC)
        return self.IPSC


def main():
    d1 = DFRSpiking()
    d1.import_images()
    d1.apply_encoding()
    d1.assign_ts()
    d1.assign_IPSC()


if __name__ == '__main__':
    main()
