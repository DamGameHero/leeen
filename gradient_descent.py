import matplotlib.pyplot as plt
import numpy as np
import sys


def get_data(args):
    file = "data.csv"
    if len(args) < 2:
        print("No arguments given. Try with \"data.csv\".")
    else:
        file = args[1]
    try:
        data = np.genfromtxt(file, delimiter=",", names=True)
    except Exception as e:
        print("Can't extract data from {}.".format(file))
        print(e.__doc__)
        sys.exit(0)
    return data


# speed up gradient descent
def feature_scaling(xAverage, xRange, data):
    return (data['km'] - xAverage) / xRange


def cost(theta0, theta1, xScaled, data):
    m = data.size
    hypothesis = theta0 + theta1 * xScaled
    targetVar = data['price']
    return 1 / (2 * m) * np.sum(np.power((hypothesis - targetVar), 2))


# theta0 derivative of cost()
def theta0_calc(theta0, theta1, lRate, xScaled, data):
    m = data.size
    hypothesis = theta0 + theta1 * xScaled
    targetVar = data['price']
    return theta0 - (lRate / m) * (np.sum((hypothesis - targetVar)))


# theta1 derivative of cost()
def theta1_calc(theta0, theta1, lRate, xScaled, data):
    m = data.size
    hypothesis = theta0 + theta1 * xScaled
    targetVar = data['price']
    return theta1 - (lRate / m) * (np.sum((hypothesis - targetVar) * xScaled))


# Sum of Squares Explained
def ssr(theta0, theta1, xScaled, data):
    hypothesis = theta0 + theta1 * xScaled
    targetVar = data['price']
    return np.sum(np.power(hypothesis - targetVar, 2))


# Sum of Squares Total
def sst(data):
    yAverage = np.average(data['price'])
    targetVar = data['price']
    return np.sum(np.power(yAverage - targetVar, 2))


def results_generation(theta0, theta1, data, turn, costs):
    try:
        np.savetxt("thetas.csv", [[theta0, theta1]], delimiter=',')
    except Exception as e:
        print("Can't open thetas.csv")
        print(e.__doc__)
    plt.figure(2)
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (euros)')
    plt.title('Data Visualization')
    plt.plot(data['km'], data['price'], "ro")
    plt.figure(3)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost Function')
    plt.title('Data Visualization')
    plt.plot(np.arange(turn + 1), costs)
    plt.figure(1)
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (euros)')
    plt.title('Linear Regression')
    plt.plot(data['km'], data['price'], "r+")
    plt.plot(data['km'], np.add(np.multiply(theta1, data['km']), theta0))
    plt.show()


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
    data = get_data(sys.argv)
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

    r_squared = 1 - (ssr(theta0, theta1, xScaled, data) / sst(data))
    tmp_theta0 = theta0
    tmp_theta1 = theta1
    theta0 = tmp_theta0 + tmp_theta1 * ((-1 * xAverage) / xRange)
    theta1 = tmp_theta0 + tmp_theta1 * ((1 - xAverage) / xRange) - theta0
    print("Number of iterations to perform gradient descent = ", turn)
    print("Sample size = ", data.size)
    print("X Average = ", xAverage)
    print("X Range = ", xRange)
    print("Y Average = ", np.average(data['price']))
    print("Y Range = ", np.amax(data['price']) - np.amin(data['price']))
    print("theta0 = ", theta0, "\ntheta1 =", theta1)
    print("Coefficient of determination R2 = ", r_squared)
    print("(R2 varies between 0 and 1. Close to 0, the predictive power of the model is weak. Close to 1, the predictive power of the model is strong.)")
    results_generation(theta0, theta1, data, turn, costs)
