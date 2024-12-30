import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
def load_data(path = ""):
    return np.loadtxt(path + "X_train.txt", delimiter=','), np.loadtxt(path + "y_train.txt", delimiter=','), np.loadtxt(path + "X_val.txt", delimiter=','), np.loadtxt(path + "y_val.txt", delimiter=',')

def std_scaler(train, test):
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

    mean_vector = np.mean(train, axis=0)
    std_dev_vector = np.std(train, axis=0)
    std_dev_vector[std_dev_vector == 0] = 1 
    train_normalized = (train - mean_vector)*(1/(std_dev_vector)) 
    test_normalized = (test - mean_vector)*(1/(std_dev_vector)) 
    return train_normalized, test_normalized



def classification_error(X, y, theta):
    y_score = np.dot(X, theta)
    y_pred = np.where(y_score > 0, 1, -1)
    return np.sum(y != y_pred)/(len(y_pred))

def sigmoid(v):
    return (1 / (1 + np.exp(-v)))

def log_likelihood(theta, X, Y):
    n = X.shape[0]
    
    z = X @ theta
    
    logistic_loss = (1/n) * np.sum(np.logaddexp(0, -Y * z))
    return logistic_loss

'''
21. Complete the f objective function in the skeleton code, 
which computes the objective function for Jlogistic(w). 
(Hint: you may get numerical overflow when computing the exponential literally, e.g. try e1000 in Numpy. 
Make sure to read about the log-sum-exp trick and use the numpy function logaddexp to get accurate calculations 
and to prevent overflow.
'''
def f_objective(theta, X, Y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    
    n = X.shape[0]
    
    z = X @ theta
    
    logistic_loss = (1/n) * np.sum(np.logaddexp(0, -Y * z))
    
    regularization_loss = l2_param * (theta.T @ theta)
    
    return logistic_loss + regularization_loss


'''
22. Complete the fit logistic regression functionintheskeleton code using the minimize function from scipy.optimize. 
Use this function to train a model on the provided data. 
Make sure to take the appropriate preprocessing steps, such as standardizing the data and adding a column for the bias term.
'''
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    theta = np.zeros(X.shape[1])
    f = lambda i : objective_function(i, X, y, l2_param)
    
    
    optimizer_obj = minimize(f, theta)
    return optimizer_obj.x

def plot_lambda_reg_vs_log_likelihood(X_train, y_train, X_test, y_test, lambda_reg = [1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]):
    log_likelihoods = []
    for l in lambda_reg:
        theta = fit_logistic_reg(X_train, y_train, f_objective, l)
        log_likelihoods.append(log_likelihood(theta, X_test, y_test))
    plt.figure(figsize=(10, 10))
    plt.plot([np.log10(i) for i in lambda_reg], log_likelihoods, 'bo-', label = 'log-lambda values vs log likelihoods')
    plt.xlabel('log10-lambda')
    plt.ylabel('NLL')
    plt.show()
    
def plot_predicted_prob_calibration(theta, X, y, bins = 5):
    y_pred_prob = sigmoid(X @ theta)
    coupled_mat = np.column_stack((y, y_pred_prob))
    coupled_mat = coupled_mat[coupled_mat[:, 1].argsort()[::-1]]
    range_start = 0
    range_end = 0
    probability_ranges = []
    percentage_positive_samples = []
    for i in np.linspace(0, coupled_mat.shape[0], bins, dtype=int):
        if i == range_end:
            continue
        range_start = range_end
        range_end = i
        mat = coupled_mat[range_start:range_end, :]
        probability_ranges.append(f"{np.min(mat[:, 1]):.2f} to {np.max(mat[:, 1]):.2f}")
        percentage_positive_samples.append((np.sum(mat[:, 0] == 1)) / mat.shape[0])
    
    plt.figure(figsize=(10, 10))
    plt.plot(probability_ranges, percentage_positive_samples, 'ro-', label='score ranges vs classification erros')
    plt.xlabel("predicted probability ranges")
    plt.ylabel("positive examples ratio")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()
    
    return 
def __main__():
    X_train, y_train, X_val, y_val = load_data()
    
    X_train, X_val = std_scaler(X_train, X_val)
    
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
    
    y_train = np.where(y_train == 0, -1, y_train)
    y_val = np.where(y_val == 0, -1, y_val)
    
    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'X_val.shape: {X_val.shape}')
    print(f'y_val.shape: {y_val.shape}')
    
    theta = fit_logistic_reg(X_train, y_train, f_objective, 0.01)

    print(f'theta.shape: {theta.shape}')
    print(f'classification error - training set: {classification_error(X_train, y_train, theta)}')
    print(f'classification error - validation set: {classification_error(X_val, y_val, theta)}')
    
    plot_lambda_reg_vs_log_likelihood(X_train, y_train, X_val, y_val)
    
    plot_predicted_prob_calibration(X= X_val, y= y_val, theta= theta, bins= 5)  
__main__()
