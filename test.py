import matplotlib.pyplot as plt
import numpy as np
import sys


def get_data():
    try:
        data = np.genfromtxt("data.csv", delimiter=",", names=True)
    except Exception as e:
        print("Can't extract data from data.csv.")
        print(e.__doc__)
        sys.exit(0)
    return data


# speed up gradient descent
def feature_scaling(xAverage, xRange, data):
    return np.divide(np.subtract(data['km'], xAverage), xRange)


def cost(theta0, theta1, xScaled, data):
    m = data.size
    hypothesis = np.add(theta0, np.multiply(theta1, xScaled))
    targetVar = data['price']
    return (1 / (2 * m)) * np.sum(np.power(np.subtract(hypothesis, targetVar), 2))


# theta0 derivative of cost()
def theta0_calc(theta0, theta1, lRate, xScaled, data):
    m = data.size
    hypothesis = np.add(theta0, np.multiply(theta1, xScaled))
    targetVar = data['price']
    return theta0 - (lRate / m) * (np.sum(np.subtract(hypothesis, targetVar)))


# theta1 derivative of cost()
def theta1_calc(theta0, theta1, lRate, xScaled, data):
    m = data.size
    hypothesis = np.add(theta0, np.multiply(theta1, xScaled))
    targetVar = data['price']
    return theta1 - (lRate / m) * (np.sum(np.multiply(np.subtract(hypothesis, targetVar), xScaled)))


if __name__ == '__main__':
    # Vairables Initialization
    turn = 0
    theta0 = 0.0
    theta1 = 0.0
    lRate = 1
    converge = 1000000
    costs = []
    tmp_cost = 0
    new_cost = 0

    # Data Preprocessing
    data = get_data()
    xAverage = np.average(data['km'])
    xMax = np.amax(data['km'])
    xMin = np.amin(data['km'])
    xRange = xMax - xMin
    xScaled = feature_scaling(xAverage, xRange, data)
    tmp_cost = cost(theta0, theta1, xScaled, data)
    costs.append(tmp_cost)

    # Gradient Descent
    while converge > 0.00000001:
        tmp_theta0 = theta0
        tmp_theta1 = theta1
        theta0 = theta0_calc(tmp_theta0, tmp_theta1, lRate, xScaled, data)
        theta1 = theta1_calc(tmp_theta0, tmp_theta1, lRate, xScaled, data)
        new_cost = cost(theta0, theta1, xScaled, data)
        costs.append(new_cost)
        converge = np.abs(tmp_cost - new_cost)
        tmp_cost = new_cost
        turn += 1

    # print("size = ", data.size)
    # print("xAverage = ", xAverage)
    # print("xRange = ", xRange)
    # print("turn = ", turn)
    tmp_theta0 = theta0
    tmp_theta1 = theta1
    theta0 = tmp_theta0 + tmp_theta1 * ((-1 * xAverage) / xRange)
    theta1 = tmp_theta0 + tmp_theta1 * ((1 - xAverage) / xRange) - theta0
    # print("theta0 = ", theta0, "\ntheta1f =", theta1)
    try:
        np.savetxt("thetas.csv", [[theta0, theta1]], delimiter=',')
    except Exception as e:
        print("Can't open thetas.csv")
        print(e.__doc__)
    plt.plot(data['km'], data['price'], "ro")
    plt.plot(data['km'], np.add(np.multiply(theta1, data['km']), theta0))
    # plt.plot(np.arange(turn + 1), costs)
    plt.show()
