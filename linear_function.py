import numpy as np
import sys


if __name__ == '__main__':
    try:
        theta = np.genfromtxt("thetas.csv", delimiter=",")
    except Exception as e:
        print("Can't extract data from thetas.csv. Thetas will be set with 0")
        print("Error Type : ", e.__doc__)
        theta = [0, 0]

    try:
        mileage = np.float64(input("Please enter a mileage : "))
        cost = theta[0] + theta[1] * mileage
        print("This car worth {0:.2f} euro".format(cost))
    except Exception as e:
        print("Incorrect value")
        print("Error Type : ", e.__doc__)
