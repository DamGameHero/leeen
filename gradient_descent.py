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


def data_preprocessing(data):
    xStats = {'xAverage': np.average(data['km'])}
    xStats['xMax'] = np.amax(data['km'])
    xStats['xMin'] = np.amin(data['km'])
    xStats['xRange'] = xStats['xMax'] - xStats['xMin']
    xStats['xSize'] = data.size
    xScaled = feature_scaling(xStats['xAverage'], xStats['xRange'], data)
    y = data['price']
    return xScaled, y, xStats


# speed up gradient descent
def feature_scaling(xAverage, xRange, data):
    return (data['km'] - xAverage) / xRange


def cost(theta0, theta1, x, y, xStats):
    m = xStats['xSize']
    hypothesis = theta0 + theta1 * x
    return 1 / (2 * m) * np.sum(np.power((hypothesis - y), 2))


# theta0 derivative of cost()
def theta0_calc(theta0, theta1, lRate, x, y, xStats):
    m = xStats['xSize']
    hypothesis = theta0 + theta1 * x
    return theta0 - (lRate / m) * (np.sum((hypothesis - y)))


# theta1 derivative of cost()
def theta1_calc(theta0, theta1, lRate, x, y, xStats):
    m = xStats['xSize']
    hypothesis = theta0 + theta1 * x
    return theta1 - (lRate / m) * (np.sum((hypothesis - y) * x))


# Sum of Squares Explained
def ssr(theta0, theta1, x, y, xStats):
    hypothesis = theta0 + theta1 * x
    return np.sum(np.power(hypothesis - y, 2))


# Sum of Squares Total
def sst(y):
    yAverage = np.average(y)
    return np.sum(np.power(yAverage - y, 2))


def r_squared_calc(theta0, theta1, x, y, xStats):
    return 1 - (ssr(theta0, theta1, x, y, xStats) / sst(y))


def display_results(results, xStats, y):
    print(
            "Number of iterations to perform gradient descent = ",
            results['turns'])
    print("Sample size = ", xStats['xSize'])
    print("X Average = ", xStats['xAverage'])
    print("X Range = ", xStats['xRange'])
    print("Y Average = ", np.average(y))
    print("Y Range = ", np.amax(y) - np.amin(y))
    print("theta0 = ", results['theta0'], "\ntheta1 =", results['theta1'])
    print("Coefficient of determination R2 = ", results['r_squared'])
    print(
            "(R2 varies between 0 and 1. Close to 0, the predictive power of "
            "the model is weak. Close to 1, "
            "the predictive power of the model is strong.)")


def results_generation(results, data):
    try:
        np.savetxt(
                "thetas.csv",
                [[results['theta0'], results['theta1']]],
                delimiter=',')
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
    plt.plot(np.arange(results['turns'] + 1), results['costs'])
    plt.figure(1)
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (euros)')
    plt.title('Linear Regression')
    plt.plot(data['km'], data['price'], "r+")
    plt.plot(
            data['km'],
            np.add(
                np.multiply(results['theta1'], data['km']),
                results['theta0']))
    plt.show()


def gradient_descent(x, y, xStats):

    # Vairables Initialization
    turn = 0
    theta0 = 0.0
    theta1 = 0.0
    lRate = 1
    converge = 1000000
    costs = []
    tmp_cost = 0
    new_cost = 0
    tmp_cost = cost(theta0, theta1, x, y, xStats)
    costs.append(tmp_cost)

    # Gradient Descent Processing
    while converge > 0.00000001:
        tmp_theta0 = theta0
        tmp_theta1 = theta1
        theta0 = theta0_calc(tmp_theta0, tmp_theta1, lRate, x, y, xStats)
        theta1 = theta1_calc(tmp_theta0, tmp_theta1, lRate, x, y, xStats)
        new_cost = cost(theta0, theta1, x, y, xStats)
        costs.append(new_cost)
        converge = np.abs(tmp_cost - new_cost)
        tmp_cost = new_cost
        turn += 1

    # Results Calc
    r_squared = r_squared_calc(theta0, theta1, x, y, xStats)
    tmp_theta0 = theta0
    tmp_theta1 = theta1
    theta0 = (
            tmp_theta0
            + tmp_theta1 * ((-1 * xStats['xAverage']) / xStats['xRange']))
    theta1 = (
            tmp_theta0
            + tmp_theta1 * ((1 - xStats['xAverage']) / xStats['xRange'])
            - theta0)
    results = {'turns': turn}
    results['costs'] = costs
    results['theta0'] = theta0
    results['theta1'] = theta1
    results['r_squared'] = r_squared
    return results


def main():
    # Get Data
    data = get_data(sys.argv)

    # Data Preprocessing
    x, y, xStats = data_preprocessing(data)

    # Gradient Descent Process
    results = gradient_descent(x, y, xStats)

    # Displaying Results
    display_results(results, xStats, y)

    # Results Generation
    results_generation(results, data)


if __name__ == '__main__':
    main()
