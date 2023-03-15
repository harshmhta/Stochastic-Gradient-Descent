import math
import random
import numpy as np


def bgd_l2(data, y, w, eta, delta, lam, num_iter):

    history_fw = []

    for a in range(num_iter):
        wa = np.matmul(data, w)
        gdel = np.where(
            y >= wa + delta,
            (y - wa - delta) ** 2,
            np.where((abs(y - wa) < delta), 0, (y - wa + delta) ** 2),
        )
        intgrad = np.where(
            y >= wx + delta,
            (-2) * (y - wa - delta),
            np.where((abs(y - wa) < delta), 0, (-2) * (y - wa + delta)),
        )
        w = w - eta * (np.matmul(intgrad, data) / len(data) + 2 * lam * w)
        history_fw.append((np.sum(gdel) / len(data)) + (lam * np.dot(w, w)))
    new_w = w
    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    
    history_fw = []

    if i == -1:
        for h in range(num_iter):
            eta = eta / (math.sqrt(h + 1))
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

        new_w = w

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
        new_w = w

    return new_w, history_fw
