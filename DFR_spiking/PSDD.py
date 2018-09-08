import numpy as np
import math
import matplotlib.pyplot as plt


# Convert spikes to analog signals, like DAC
def PSDD(TS):
    tsp1 = TS
    t = np.arange(0, 152, 2)
    Ipsc1 = np.zeros(len(t))
    Ipsc2 = np.zeros(len(t))
    Ipsc3 = np.zeros(len(t))
    ipsc = np.zeros(len(t))

    for i in range(len(t)):
        Ipsc1[i] = (math.exp(-(t[i] - tsp1[0]) / 10) - math.exp(-(t[i] - tsp1[0]) / 2.5))\
                   * np.heaviside(t[i] - tsp1[0], 0.5)
        Ipsc2[i] = (math.exp(-(t[i] - tsp1[1]) / 10) - math.exp(-(t[i] - tsp1[1]) / 2.5))\
                   * np.heaviside(t[i] - tsp1[1], 0.5)
        Ipsc3[i] = (math.exp(-(t[i] - tsp1[2]) / 10) - math.exp(-(t[i] - tsp1[2]) / 2.5))\
                   * np.heaviside(t[i] - tsp1[2], 0.5)

    # for i in range(len(t)):
    #     ipsc[i] = Ipsc1[i] + Ipsc2[i] + Ipsc3[i]
    ipsc = Ipsc1 + Ipsc2 + Ipsc3

    return ipsc


def main():
    input = np.array((28.3333, 85, 113.3333))
    ans = PSDD(input)
    plt.plot(ans)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
