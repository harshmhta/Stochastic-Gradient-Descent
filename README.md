# Stochastic Gradient Descent with L2 Regularization

## Overview

In this project, we implement a stochastic gradient descent algorithm with L2 regularization to optimize a linear regression model. The algorithm uses a fixed learning rate, which is the square root of the iteration number, and allows for either random data point selection or specific data point selection.

The project has two Python files: Problem5.py and Problem5Plot.py. The former contains the implementation of the SGD algorithm with L2 regularization, while the latter generates plots for the objective function history for the different settings.

## Usage

### Dependencies

The following dependencies are required to run the project:

Python 3

NumPy

Matplotlib

## Running the code

Clone the repository.

Open a terminal and navigate to the project directory.

Run python Problem5.py to run the SGD algorithm and store the objective function history data in obj_func_hist.npz.

Run python Problem5Plot.py to generate the plots for the different settings. The plots will be saved in the plots directory.

## Files

### Problem5.py

This file contains the implementation of the SGD algorithm with L2 regularization. It has the following function:

### sgd_l2(data, y, w, eta, delta, lam, num_iter, i)

This function implements the stochastic gradient descent algorithm with L2 regularization. It takes the following arguments:

data : a numpy array of shape (n, d) representing the data points

y : a numpy array of shape (n,) representing the labels

w : a numpy array of shape (d,) representing the initial weight vector

eta : a float representing the fixed learning rate (square root of iteration number)

delta : a float representing the delta value for the Huber loss function

lam : a float representing the L2 regularization coefficient

num_iter : an integer representing the number of iterations to run the SGD algorithm

i : an integer representing the index of the data point to use for the SGD algorithm (if -1, then use random data point selection)

The function returns the following:

w_hist : a numpy array of shape (num_iter+1, d) representing the weight vectors at each iteration

obj_func_hist : a numpy array of shape (num_iter+1,) representing the objective function value at each iteration

### Problem5Plot.py

This file generates plots for the objective function history for the different settings. It has the following function:

#### generate_plots()

This function generates plots for the objective function history for the different settings. It uses the sgd_l2 function from Problem5.py to run the SGD algorithm for the different settings and saves the objective function history data in obj_func_hist.npz. The plots are saved in the plots directory.

## Data

The sample data for this project has 100 data points (xi,yi) in a 100 × 2 numpy array. We add a column of all ones to data to handle the intercept term, making the data a 100 × 3 numpy array. The data is stored in data.npy.

## Outputs

The output of the project is a set of plots showing the objective function history for the different settings.

## Conclusion

We implemented a stochastic gradient descent algorithm with L2 regularization to optimize a linear regression model. The algorithm used a fixed learning rate, which is the square root of the iteration number, and allowed for either random data point selection or specific data point selection. We also generated plots to visualize the objective function history for different settings of the algorithm.

The implementation of the algorithm and the visualization of the results provide a good foundation for further exploration and experimentation with stochastic gradient descent and L2 regularization. This project can be expanded upon to explore different learning rates, regularization coefficients, and loss functions to further optimize the linear regression model.

## Academic Integrity Statement
Please note that all work included in this project is the original work of the author, and any external sources or references have been properly cited and credited. It is strictly prohibited to copy, reproduce, or use any part of this work without permission from the author.

If you choose to use any part of this work as a reference or resource, you are responsible for ensuring that you do not plagiarize or violate any academic integrity policies or guidelines. The author of this work cannot be held liable for any legal or academic consequences resulting from the misuse or misappropriation of this work.

In summary, any unauthorized copying or use of this work may result in serious consequences, including but not limited to academic penalties, legal action, and damage to personal and professional reputation. Therefore, please use this work only as a reference and always ensure that you properly cite and attribute any sources or references used.
