import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

### Assignment Owner: Tian Wang, Marylou Gabrié


#######################################
### 1. Feature Normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size(num_instances, num_features)
        test - test set, a 2D numpy array of size(num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # for every feature the affine transformation will be
    # subract the mean and divide by the range
    min_vector = np.min(train, axis=0)
    range_vector = np.max(train, axis=0) - np.min(train, axis=0)
    range_vector[range_vector == 0] = 1 # this handles the case where the min = max, i.e. the features are constant and range = 0. This way we avoid 0/0 division when the features are constant. train - min_vector  = 0 as well and the feature is thus ignored.
    train_normalized = (train - min_vector)*(1/(range_vector)) 
    test_normalized = (test - min_vector)*(1/(range_vector)) 
    return train_normalized, test_normalized



#######################################
### 5. The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D array of size(num_features)

    Returns:
        loss - the average square loss, scalar
    """
    # number of training samples
    m = len(y)
    J = (1/m)*(((X @ theta) - y).T @ ((X @ theta) - y))
    # convert 1x1 numpy matrix to a scalar value
    return J.item()



#######################################
### 6. The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss(as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    m = len(y)
    grad_J = (2/m) * (((X.T @ X) @ theta) - (X.T @ y))
    return grad_J


#######################################
### 7. Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
(e_1 =(1,0,0,...,0), e_2 =(0,1,0,...,0), ..., e_d =(0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
(J(theta + epsilon * e_i) - J(theta - epsilon * e_i)) /(2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    

    E = np.eye(num_features)
    
    theta_plus_epsilon_e = theta + (epsilon*E)
    theta_minus_epsilon_e = theta - (epsilon*E)
    
    # trying out list comprehension for more optimized compute
    approx_grad = (1/(2*epsilon)) * (np.array([compute_square_loss_gradient(X, y, i) for i in theta_plus_epsilon_e]) - np.array([compute_square_loss_gradient(X, y, i) for i in theta_minus_epsilon_e]))
    return tolerance >= np.linalg.norm(approx_grad - true_gradient) 

#######################################
### Generic gradient checker
# optional
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, 
                             epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO



#######################################
### 8. Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array,(num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step + 1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  #Initialize loss_hist
    theta = np.zeros(num_features)  #Initialize theta
    loss_hist[0] = 1e64
    # theta gets updated as theta := theta - alpha*grad(Jtheta)
    for i in range(1, num_step + 1):
        if (grad_check == True):
            error = np.sum(grad_checker(X, y, theta)) 
            if error == True:
                print("at step i = " + str(i) + " we find that gradient has failed to compute successfully. \
                    exiting program")
                return
        theta = theta - alpha*(compute_square_loss_gradient(X, y, theta))
        theta_hist[i:] = theta
        loss_hist[i] = compute_square_loss(X, y, theta)
        # print("loss: " + str(loss_hist[i]))
    return theta_hist, loss_hist
    
#######################################
# 9. Now let’s experiment with the step size. Note that if the step size is too large, gradient descent may not converge. Starting with a step-size of 0.1, try various different fixed step sizes to see which converges most quickly and/or which diverge. As a minimum, try step sizes 0.5, 0.1, .05, and .01. Plot the average square loss on the training set as a function of the number of steps for each step size. Briefly summarize your findings
def plot_training_loss_vs_iterations_for_multiple_step_sizes(X_train, y_train, step_sizes = [0.5, 0.1, 0.05, 0.01], max_iterations = 1000, loss_cap = 20):
    plt.figure(figsize=(10, 10))
    for step_size in step_sizes:
        theta_hist, loss_hist = batch_grad_descent(X_train, y_train, alpha= step_size, num_step= max_iterations, grad_check= False)
        loss_hist[loss_hist > loss_cap] = loss_cap
        plt.plot((range(0, len(loss_hist))), loss_hist, label = "step_size = " + str(step_size))
    plt.xlabel("iterations")
    plt.ylabel("training loss")
    plt.legend()
    plt.show()

#######################################
# 10. For the learning rate you selected above, plot the average test loss as a function of the iterations. You should observe overfitting: the test error first decreases and then increases. 
def plot_test_loss_vs_iterations(X_train, X_test, y_train, y_test, step_size = 0.05, \
    max_iterations = 1000, loss_cap = 20):
    plt.figure(figsize=(10, 10))
    theta_hist, training_loss_hist = batch_grad_descent(X_train, y_train, alpha= step_size, num_step= max_iterations, \
        grad_check= False)
    testing_loss_hist = np.zeros(max_iterations + 1)
    for i in range(max_iterations + 1):
        testing_loss_hist[i] = compute_square_loss(X_test, y_test, theta= theta_hist[i])
    testing_loss_hist[testing_loss_hist > loss_cap] = loss_cap
    plt.plot((range(0, max_iterations + 1)), testing_loss_hist, label = "step_size = " + str(step_size))
    plt.xlabel("iterations")
    plt.ylabel("avg test loss")
    plt.legend()
    plt.show()
#######################################
### 12. The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    m = len(y)
    grad_J = (2/m) * (((X.T @ X) @ theta) - (X.T @ y)) + 2*lambda_reg*theta
    return grad_J


#######################################
### 13. Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    for i in range(1, num_step + 1):
        theta = theta - alpha*(compute_regularized_square_loss_gradient(X, y, theta, lambda_reg))
        theta_hist[i:] = theta
        loss_hist[i] = compute_square_loss(X, y, theta) + lambda_reg * ((theta.T @ theta).item())
        # print("loss: " + str(loss_hist[i]))
    return theta_hist, loss_hist



#######################################
# 14.  Choosing a reasonable step-size, plot training average square loss and the test average square loss (just the average square loss part, without the regularization, in each case) as a function of the training iterations for various values of λ. What do you notice in terms of overfitting?

def plot_loss_vs_iterations_for_multiple_lambda_reg(X_train, X_test, y_train, y_test, step_size = 0.05, lambda_regs = [1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100], max_iterations = 1000, loss_cap = 20):
    plt.figure(figsize=(10, 10))
    colors = ['g', 'b', 'r', 'y', 'c', 'm', 'k']
    color_itr = 0
    for lambda_reg in lambda_regs:
        theta_hist, training_loss_hist = regularized_grad_descent(X_train, y_train, alpha= step_size, num_step= max_iterations, lambda_reg=lambda_reg)
        
        train_avg_sq_loss = np.array([compute_square_loss(X_train, y_train, theta) for theta in theta_hist])
        train_avg_sq_loss[train_avg_sq_loss > loss_cap] = loss_cap
        
        test_avg_sq_loss = np.array([compute_square_loss(X_test, y_test, theta) for theta in theta_hist])
        test_avg_sq_loss[test_avg_sq_loss > loss_cap] = loss_cap
        
        plt.plot(range(0, max_iterations + 1), train_avg_sq_loss, label = "training loss: lambda_reg = " + str(lambda_reg), color = colors[color_itr%(len(colors))])
        plt.plot(range(0, max_iterations + 1), test_avg_sq_loss, label = "testing loss: lambda_reg = " + str(lambda_reg), marker = '*', color = colors[color_itr%(len(colors))], linestyle = '--')
        color_itr += 1
    plt.xlabel("iterations")
    plt.ylabel("avg sq loss")
    plt.legend()
    plt.show()

#######################################
# 15. Plot the training average square loss and the test average square loss at the end of training as a function of λ. You may want to have log(λ) on the x-axis rather than λ. Which value of λ would you choose ?

def plot_loss_vs_lambda_reg(X_train, X_test, y_train, y_test, lambda_reg = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1e0, 10, 100, 0], step_size = 0.05, iterations = 1000):
    training_loss = []
    testing_loss = []
    log_lambda_reg = []
    for i in lambda_reg:
        theta_hist, loss_hist = regularized_grad_descent(X_train, y_train, step_size, i, iterations)
        training_loss.append(compute_square_loss(X_train, y_train, theta_hist[-1]))
        if(training_loss[-1] > 20):
            training_loss[-1] = 20
        testing_loss.append(compute_square_loss(X_test, y_test, theta_hist[-1]))
        if(testing_loss[-1] > 20):
            testing_loss[-1] = 20
        log_lambda_reg.append(np.log10(i))
    
    plt.figure(figsize=(10, 10))
    plt.plot(log_lambda_reg, training_loss, label = "Training loss")
    plt.plot(log_lambda_reg, testing_loss, label = "Testing loss")
    plt.ylabel("avg square loss")
    plt.xlabel("log10(lambda_reg)")
    plt.legend()
    plt.show() 
    
def load_data():
    #Loading the dataset
    #print('loading the dataset')

    df = pd.read_csv('ridge_regression_dataset.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]
    
    #print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)
    #print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)

    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    
    return X_train, y_train, X_test, y_test



# -----------------------------------------------------------------
# load the data
X_train, y_train, X_test, y_test = load_data()
    
# 9.
plot_training_loss_vs_iterations_for_multiple_step_sizes(X_train, y_train, max_iterations= 5000, loss_cap=10)
# Conclusion: the training loss seems to converge best and fastest for step_size = 0.5

#10. the training loss seems to converge best and fastest for step_size = 0.5
plot_test_loss_vs_iterations(X_train, X_test, y_train, y_test, step_size=0.05, max_iterations=1000, loss_cap=20)
# Conclusion: for step_size = 0.05; test loss dips till about 200 iterations and then starts increasing again due to overfitting

#14. plot loss vs iterations for multiple lambda regs
plot_loss_vs_iterations_for_multiple_lambda_reg(X_train, X_test, y_train, y_test, step_size=0.05, lambda_regs= [1e-5, 1e-3, 1e-2, 1e-1, 1, 10, 100], loss_cap=10, max_iterations=3000)
# Conclusion:  1e-7, 1e-5, 1e-3, 0 behave quite similarly. indicating that the value of lambda is too small to make a significant different to the model convergence. Model starts to overfit after 200 iterations or so, but still performs very well on both testing and training data. 

# overall 1e-2 performs the best but has slight overfitting 

# for lambda_reg = 0.1 we see that the model experiences higher loss on the training set, however the testing loss is always lower than the training loss indicating that the model never overfits.

# it is only at about 1500 iterations do we see that the error due overfitting exceed the model with lambda_reg = 0.1 (which never overfits)

#15. 
plot_loss_vs_lambda_reg(X_train, X_test, y_train, y_test, lambda_reg = [1e-7, 0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 0], step_size= 0.05, iterations= 3000)
# 1e-2