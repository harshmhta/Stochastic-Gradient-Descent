import random
import numpy as np
import matplotlib.pyplot as plt


def bgd_l2_plot(
    data, y, w, eta, delta, lam, num_iter
):  
    
    history_fw = []
    
    for i in range(num_iter):
        wa = np.matmul(data, w)
        gdel = np.where(
            y >= wa + delta,
            (y - wa - delta) ** 2,
            np.where((abs(y - wa) < delta), 0, (y - wa + delta) ** 2),
        )
        intgrad = np.where(
            y >= wa + delta,
            (-2) * (y - wa - delta),
            np.where((abs(y - wa) < delta), 0, (-2) * (y - wa + delta)),
        )
        w = w - eta * (np.matmul(intgrad, data) / len(data) + 2 * lam * w)
        history_fw.append((np.sum(gdel) / len(data)) + (lam * np.dot(w, w)))

    plt.plot(history_fw)
    plt.xlabel("Number of iterations")
    plt.ylabel("Objective function history")
    plt.title(
        "GD: eta = "
        + str(eta)
        + ", delta = "
        + str(delta)
        + ", lambda = "
        + str(lam)
        + ", num_iterations = "
        + str(num_iter)
    )
    plt.show()


def sgd_l2_plot(
    data, y, w, eta, delta, lam, num_iter, i=-1
):  
    
    history_fw = []

    if i == -1:
        for j in range(num_iter):
            eta = eta / (math.sqrt(j + 1))
            wa = np.matmul(data, w)
            gdel = np.where(
                y >= wa + delta,
                (y - wa - delta) ** 2,
                np.where((abs(y - wa) < delta), 0, (y - wa + delta) ** 2),
            )
            intgrad = np.where(
                y >= wa + delta,
                (-2) * (y - wa - delta),
                np.where((abs(y - wa) < delta), 0, (-2) * (y - wa + delta)),
            )
            w = w - eta * (np.matmul(intgrad, data) / len(data) + 2 * lam * w)
            history_fw.append((np.sum(gdel) / len(data)) + (lam * np.dot(w, w)))

    else:
        eta = eta / (math.sqrt(i + 1))
        wa = np.matmul(data, w)
        gdel = np.where(
            y >= wa + delta,
            (y - wa - delta) ** 2,
            np.where((abs(y - wa) < delta), 0, (y - wa + delta) ** 2),
        )
        intgrad = np.where(
            y >= wa + delta,
            (-2) * (y - wa - delta),
            np.where((abs(y - wa) < delta), 0, (-2) * (y - wa + delta)),
        )
        w = w - eta * (np.matmul(intgrad, data) / len(data) + 2 * lam * w)
        history_fw.append((np.sum(gdel) / len(data)) + (lam * np.dot(w, w)))

    plt.plot(history_fw)
    plt.xlabel("Number of iterations")
    plt.ylabel("Objective function history")
    plt.title(
        "SGD: eta = "
        + str(eta)
        + ", delta = "
        + str(delta)
        + ", lambda = "
        + str(lam)
        + ", num_iterations = "
        + str(num_iter)
    )
    plt.show()


if __name__ == "__main__":
    # Put the code for the plots here, you can use different functions for each
    # part
    data = np.load("data.npy")
    arr = np.full((data.shape[0], 1), 1)
    data = np.append(data, arr, axis = 1)
    y = data[:, 1]
    data = data[:, [0, 2]]
    w = np.array([0, 0])

    bgd_l2_plot(data, y, w, 0.05, 0.1, 0.001, 50)
    bgd_l2_plot(data, y, w, 0.1, 0.01, 0.001, 50)
    bgd_l2_plot(data, y, w, 0.1, 0, 0.001, 100)
    bgd_l2_plot(data, y, w, 0.1, 0, 0, 100)

    sgd_l2_plot(data, y, w, 1, 0.1, 0.5, 800)
    sgd_l2_plot(data, y, w, 1, 0.01, 0.1, 800)
    sgd_l2_plot(data, y, w, 1, 0, 0, 40)
    sgd_l2_plot(data, y, w, 1, 0, 0, 800)
