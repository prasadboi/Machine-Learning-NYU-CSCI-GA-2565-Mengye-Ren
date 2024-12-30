import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pre_process_mnist_01():
    """
    Load the mnist datasets, selects the classes 0 and 1 
    and normalize the data.
    Args: none
    Outputs: 
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    X_mnist, y_mnist = fetch_openml('mnist_784', version=1, 
                                    return_X_y=True, as_frame=False)
    indicator_01 = (y_mnist == '0') + (y_mnist == '1')
    X_mnist_01 = X_mnist[indicator_01]
    y_mnist_01 = y_mnist[indicator_01]
    X_train, X_test, y_train, y_test = train_test_split(X_mnist_01, y_mnist_01,
                                                        test_size=0.33,
                                                        shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test)

    y_test = 2 * np.array([int(y) for y in y_test]) - 1
    y_train = 2 * np.array([int(y) for y in y_train]) - 1
    return X_train, X_test, y_train, y_test


def sub_sample(N_train, X_train, y_train):
    """
    Subsample the training data to keep only N first elements
    Args: none
    Outputs: 
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    assert N_train <= X_train.shape[0]
    return X_train[:N_train, :], y_train[:N_train]

# 18. To evaluate the quality of our model we will use the classification error, which corresponds
# to the fraction of incorrectly labeled examples. For a given sample, the classification error
# is 1 if no example was labeled correctly and 0 if all examples were perfectly labeled. Using
# the method clf.predict() from the classifier write a function that takes as input an
# SGDClassifier which we will call clf, a design matrix X and a target vector y and returns
# the classification error. You should check that your function returns the same value as
# 1 - clf.score(X, y).
def classification_error(clf, X, y):
    return np.sum(clf.predict(X) != y)/X.shape[0]

# 19. Report the test classification error achieved by the logistic regression as a function of the
# regularization parameters α (taking 10 values between 10−4 and 10−1). You should make a
# plot with α as the x-axis in log scale. For each value of α, you should repeat the experiment
# 10 times so has to finally report the mean value and the standard deviation. You should
# use plt.errorbar to plot the standard deviation as error bars
def classification_error_vs_lambda_reg(X_train, y_train, lambda_reg_lo = 1e-4, lambda_reg_hi = 1e-1):
    log_lambda_reg = []
    mean_error_list = []
    std_dev_error_list = []
    epsilon = 1e-9
    for i in np.arange(np.log10(lambda_reg_lo) + epsilon, np.log10(lambda_reg_hi) - epsilon, (np.log10(lambda_reg_hi) - np.log10(lambda_reg_lo))/10): # getting exactly 10 values between the upper and lower limit
        error_arr = np.zeros(10)
        for j in range(10):
            classifier = SGDClassifier(
                loss = 'log_loss',
                max_iter = 1000,
                tol = 1e-3,
                penalty = 'l1',
                alpha= np.power(10, i),
                learning_rate='invscaling',
                power_t = 0.5,
                verbose = 1,
                eta0 = 0.01
            )
            classifier.fit(X_train, y_train)
            
            error_arr[j] = classification_error(classifier, X_test, y_test)
        
        mean_error = np.mean(error_arr)
        std_dev = np.std(error_arr)
        log_lambda_reg.append(i)
        mean_error_list.append(mean_error)
        std_dev_error_list.append(std_dev)
        
    plt.figure(figsize=(10, 10))
    plt.errorbar(x = log_lambda_reg, y = mean_error_list, yerr=std_dev_error_list)
    plt.legend()
    plt.show()
    return

# 20. What source of randomness are we averaging over by repeating this experiment ?
# A: The randomized initialization of theta

# 21. What is the optimal value of the parameter alpha among the values you tested ?
# A. alpha = 1e(-1.3); has the mean error ~ 0.0037; has the std deviation ~ (0.0040 - 0.0034)*0.5 = 0.0003

# 22. Finally, for one run of the fit for each value of α plot the value of the fitted θ. You can access it via clf.coef , and should reshape the 764 dimensional vector to a 28×28 arrray to visualize it with plt.imshow. Defining scale = np.abs(clf.coef ).max(), you can use the following keyword arguments (cmap=plt.cm.RdBu, vmax=scale, vmin=-scale) which will set the colors nicely in the plot. You should also use a plt.colorbar() to visualize the values associated with the colors.
def plot_theta_vs_lambda_reg(X_train, y_train, lambda_reg_lo = 1e-4, lambda_reg_hi = 1e-1):
    plt.figure(figsize=(20, 20))
    epsilon = 1e-9
    idx = 0
    for i in np.arange(np.log10(lambda_reg_lo) + epsilon, np.log10(lambda_reg_hi) - epsilon, (np.log10(lambda_reg_hi) - np.log10(lambda_reg_lo))/10): # getting exactly 10 values between the upper and lower limit
        classifier = SGDClassifier(
            loss = 'log_loss',
            max_iter = 1000,
            tol = 1e-3,
            penalty = 'l1',
            alpha= np.power(10, i),
            learning_rate='invscaling',
            power_t = 0.5,
            verbose = 0,
            eta0 = 0.01
        )
        classifier.fit(X_train, y_train)
        theta = classifier.coef_.reshape(28, 28)
        scale = np.abs(classifier.coef_).max()
        idx = idx+1
        plt.subplot(4, 4, idx)
        plt.title("log10(lambdareg) = " + str(i))
        plt.imshow(theta, cmap=plt.cm.RdBu, vmax=scale, vmin=-scale)
        plt.colorbar()
    plt.show()
        
X_train, X_test, y_train, y_test = pre_process_mnist_01()
X_train, y_train = sub_sample(100, X_train, y_train)

clf = SGDClassifier(loss='log_loss', max_iter=1000, 
                    tol=1e-3,
                    penalty='l1', alpha=0.01, 
                    learning_rate='invscaling', 
                    power_t=0.5,                
                    eta0=0.01,
                    verbose=1)
clf.fit(X_train, y_train)





test = classification_error(clf, X_test, y_test)
train = classification_error(clf, X_train, y_train)

print('train: ', train, end='\t')
print('test: ', test)
print('train: ', 1 - clf.score(X_train, y_train), end='\t')
print('test: ', 1 - clf.score(X_test, y_test))

X_train, y_train = sub_sample(100, X_train, y_train)

classification_error_vs_lambda_reg(X_train, y_train)
plot_theta_vs_lambda_reg(X_train, y_train, 1e-4, 1e-1)