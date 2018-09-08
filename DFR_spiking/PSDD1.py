import numpy as np
import math
import matplotlib.pyplot as plt
import LIF1


# Convert spikes to analog
def psdd1(TS):
    tsp1 = TS
    t = np.arange(0, 152, 2)
    ipsc1 = np.zeros((len(TS), len(t)))
    for l in range(len(tsp1)):
        for i in range(len(t)):
            ipsc1[l,i] = ((math.exp(-(t[i] - tsp1[l])/10)) - math.exp(-(t[i]-tsp1[l])/2.5))\
                         * np.heaviside(t[i] - tsp1[l], 0.5)
    ret = np.zeros(len(ipsc1[0]))
    for i in range(len(ipsc1[0])):
        ret[i] = ipsc1[:, i].sum()
    return ret


def main():
    infile = 'IPSC9delay30.npy'
    ipsc = np.load(infile)
    x = ipsc[0, 30, :]
    TT = LIF1.LIF1(16 * x)
    ans = psdd1(TT)
    plt.plot(ans, c=np.random.rand(3, ))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
