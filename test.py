import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data = np.genfromtxt("data.csv", delimiter = ",", names = True)
    cnt = 0
    teta0 = 0.0
    teta1 = 0.0
    lRate = 1
    cost = []
    kaverage = np.average(data['km'])
    kmax = np.amax(data['km'])
    kmin = np.amin(data['km'])
    krange = kmax - kmin
    knew = np.divide(np.subtract(data['km'], kaverage), krange)
    cost.append((1 / (2 * float(data.size))) * (np.sum(np.power(np.subtract(np.add(teta0, np.multiply(teta1, knew)), data['price']), 2))))
    while cnt < 350:
        tmp0 = teta0
        tmp1 = teta1
        teta0 = tmp0 - (lRate / float(data.size)) * (np.sum(np.subtract(np.add(tmp0, np.multiply(tmp1, knew)), data['price'])))
        teta1 = tmp1 - (lRate / float(data.size)) * (np.sum(np.multiply(np.subtract(np.add(tmp0, np.multiply(tmp1, knew)), data['price']), knew)))
        cost.append((1 / (2 * float(data.size))) * (np.sum(np.power(np.subtract(np.add(teta0, np.multiply(teta1, knew)), data['price']), 2))))
        cnt += 1

    print("lRate = ", lRate / float(data.size))
    #print("cost", cost)
    print(teta0, teta1)
    print(kaverage)
    print(krange)
    #teta0f = 8499
    #teta1f = -0.02149
    teta0f = teta0 + teta1 * ((-1 * kaverage) / krange)
    teta1f = teta0 + teta1 * ((1 - kaverage) / krange) - teta0f
    print(teta0f, teta1f)
    plt.plot(data['km'], data['price'], "ro")
    plt.plot(data['km'], np.add(np.multiply(teta1, knew), teta0))
    plt.plot(data['km'], np.add(np.multiply(teta1f, data['km']), teta0f))
    #plt.plot(np.arange(cnt + 1),cost)
    plt.show()
