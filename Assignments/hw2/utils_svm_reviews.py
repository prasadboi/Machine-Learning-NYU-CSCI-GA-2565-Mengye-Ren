import os
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import time
import random
import matplotlib.pyplot as plt


def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)


def load_and_shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "data_reviews/data/pos"
    neg_path = "data_reviews/data/neg"

    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    random.shuffle(review)
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())
    
def scale(d1, scale):
    w = {}
    for f, v in d1.items():
        w[f] = scale*v
    return w
    
def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

# 5. Write a function that converts an example (a list of words) into a sparse bag-of-words
# representation. You may find Python’s Counter class to be useful here. 
# Note that a Counter is itself a dictionary
def gen_sparse_bag_of_words(list_of_words):
    return Counter(list_of_words)

def get_predictions(X_test, W):
    return [np.sign(dotProduct(W, i)) for i in X_test]

def classification_error(X, y, W):
    y_pred = get_predictions(X, W)
    return sum(np.array(y) != np.array(y_pred))/(len(y_pred))



def pegasos_v1(X_train, y_train, lambda_reg, epochs, verbose = True, tolerance = 0.001):
    W = {}
    n = len(X_train)
    err = np.inf
    
    for t in range(1, n*epochs+1):
        if(t%epochs == 0):
            # reshuffle the data
            data = list(zip(X_train, y_train))
            random.shuffle(data)
            X_train, y_train = zip(*data)
            
        eta = 1/(lambda_reg*t)
        Xj = X_train[(t-1)%n]
        yj = y_train[(t-1)%n]
        
        margin = yj * dotProduct(W, Xj)
        
        if margin >= 1:
            for i, v in Xj.items():
                W[i] = (1 - (eta*lambda_reg))*W.get(i, 0)
        else:
            for i, v in Xj.items():
                W[i] = (1 - (eta*lambda_reg))*W.get(i, 0) + v*eta*yj
            
        
        if (t%n == 0) and (verbose == True): 
            clf_error = classification_error(X_train, y_train, W)
            print('-----------------------------------')
            print(f'epoch: {t/n}')
            print(f'classification error is: {clf_error}')
            print(f'W.size is : {len(W)}')
            if abs(err - clf_error) <= tolerance:
                break
            err = clf_error
        
    return W
        
def pegasos_v2(X_train, y_train, lambda_reg, epochs, verbose = True, tolerance = 0.001):
    W = {}
    s = 1
    n = len(X_train)
    err = np.inf
    
    for t in range(2, epochs*n + 1):
        if(t%epochs == 0):
            # reshuffle the data
            data = list(zip(X_train, y_train))
            random.shuffle(data)
            X_train, y_train = zip(*data)

        eta = 1 / (lambda_reg*t)
        Xj = X_train[(t-1)%n]
        yj = y_train[(t-1)%n]
        margin = yj * dotProduct(W, Xj) * s
        
        s = (1 - eta*lambda_reg)*s
        if margin < 1:
            increment(W, (1/s)*eta*yj, Xj)
        
        if(t%n == 0 and verbose == True):
            clf_error = classification_error(X_train, y_train, scale(W, s))
            print('-----------------------------------')
            print(f'epoch: {t/n}')
            print(f'classification error is: {clf_error}')
            print(f'W.size is : {len(W)}')
            if abs(err - clf_error) <= tolerance:
                break
            err = clf_error
    return scale(W, s)

def plot_lambda_vs_error(X_train, y_train, X_test, y_test, lambda_reg_list = [], epochs = 100):
    error_list = []
    for lambda_reg in lambda_reg_list:
        W = pegasos_v2(X_train, y_train, lambda_reg= lambda_reg, epochs= epochs, tolerance = -1, verbose=False)
        error_list.append(classification_error(X_test, y_test, W)*100)
    plt.figure(figsize=(10, 10))
    plt.plot([np.log10(i) for i in lambda_reg_list], error_list, 'bo-', label='Lambda vs %Classification Error')
    plt.xlabel("log_lambda_reg")
    plt.ylabel("% classification error")
    plt.show()
    
def plot_score_grps_vs_percentage_error(X, y, W, bins):
    scores = [dotProduct(i, W) for i in X]
    coupled_mat = np.column_stack((y, scores))
    coupled_mat = coupled_mat[coupled_mat[:, 1].argsort()[::-1]]
    range_start = 0
    range_end = 0
    score_ranges = []
    classification_errors = []
    
    for i in np.linspace(0, coupled_mat.shape[0], bins, dtype=int):
        if i == range_end:
            continue
        range_start = range_end
        range_end = i
        mat = coupled_mat[range_start:range_end, :]
        score_ranges.append(f"{np.min(mat[:, 1]):.2f} to {np.max(mat[:, 1]):.2f}")
        classification_errors.append(100 * (np.sum(mat[:, 0] != np.sign(mat[:, 1]))) / mat.shape[0])
    
    plt.figure(figsize=(10, 10))
    plt.plot(score_ranges, classification_errors, 'ro-', label='score ranges vs classification erros')
    plt.xlabel("score ranges")
    plt.ylabel("%classification error")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()

def __main__():
    
    data = load_and_shuffle_data()
    '''
    5. Write a function that converts an example (a list of words) into a sparse bag-of-words
    representation. You may find Python's Counter3 class to be useful here. Note that a Counter is itself a dictionary.
    
    A: refer function: gen_sparse_bag_of_words
    '''
    
    '''
    6. Load all the data and split it into 1500 training examples and 500 validation examples. 
    Format the training data as a list X train of dictionaries and y train as the list of corresponding 1 or-1 labels. 
    Format the test set similarly.
    '''
    print('--------------------------------------------------------------------------------------------')
    print('A6')
    X = list(map(lambda review : gen_sparse_bag_of_words(review[0 : len(review) - 1]), data))
    y = list(map(lambda review_label : review_label[-1], data))
    
    X_train = X[0:1500]
    X_test = X[1500:]
    y_train = y[0:1500]
    y_test = y[1500:]
    
    print(f'X_train: {len(X_train)} samples')
    print(f'X_test: {len(X_test)} samples')
    print(f'y_train: {len(y_train)} samples')
    print(f'y_test: {len(y_test)} samples')
    
    '''
    7. Implement the Pegasos algorithm to run on a sparse data representation. 
    The output should be a sparse weight vector w represented as a dictionary. 
    Note that our Pegasos algorithm starts at w = 0, which corresponds to an empty dictionary. 
    Terminate the algorithm when the classification error is within a tolerance of 0.001. 
    Note: With this problem, you will need to take some care to code things efficiently. 
    In particular, be aware that making copies of the weight dictionary can slow down your code significantly. 
    If you want to make a copy of your weights (e.g. for checking for convergence), make sure you don't do this more than once per epoch. 
    Also: If you normalize your data in some way, be sure not to destroy the sparsity of your data. Anything that starts as 0 should stay at 0
    '''
    print('--------------------------------------------------------------------------------------------')
    print('A7')
    begin = time.time()
    W_pegasos_v1 = pegasos_v1(X_train, y_train, lambda_reg= 0.001, epochs= 50, verbose= True, tolerance= 0.001)
    print(f'training time - pegasos_v1: {time.time() - begin}')
    print(f'training error: {classification_error(X_train, y_train, W_pegasos_v1)}')
    print(f'validation error: {classification_error(X_test, y_test, W_pegasos_v1)}')
    
    
    '''
    8. If the update is wt+1 = (1-ηtλ)wt +ηtyjxj, then verify that the Pegasos update step is equivalent to: 
        st+1 = (1-ηtλ)st 
        Wt+1 = Wt+ 1 st+1 ηtyjxj. 
    Implement the Pegasos algorithm with the (s,W) representation described above.
    '''
    print('--------------------------------------------------------------------------------------------')
    print('A8')
    begin = time.time()
    W_pegasos_v2 = pegasos_v2(X_train, y_train, lambda_reg= 0.001, epochs= 50, verbose= True, tolerance= 0.001)
    print(f'training time - pegasos_v2: {time.time() - begin}')
    print(f'training error: {classification_error(X_train, y_train, W_pegasos_v2)}')
    print(f'validation error: {classification_error(X_test, y_test, W_pegasos_v2)}')
    
    '''
    9. Run both implementations of Pegasos on the training data for a couple epochs. 
    Make sure your implementations are correct by verifying that the two approaches give essentially the same result. 
    Report on the time taken to run each approach.
    '''
    print('--------------------------------------------------------------------------------------------')
    print('A9')
    print('epochs = 2')
    begin = time.time()
    pegasos_v1(X_train, y_train, lambda_reg= 0.01, epochs= 2, verbose= False, tolerance= 0.001)
    print(f'training time - pegasos_v1: {time.time() - begin}')
    begin = time.time()
    pegasos_v2(X_train, y_train, lambda_reg= 0.01, epochs= 2, verbose= False, tolerance= 0.001)
    print(f'training time - pegasos_v2: {time.time() - begin}')
    
    print('epochs = 200. Tolerance has been disabled')
    begin = time.time()
    pegasos_v1(X_train, y_train, lambda_reg= 0.01, epochs= 200, verbose= False, tolerance= -1)
    print(f'training time - pegasos_v1: {time.time() - begin}')
    begin = time.time()
    pegasos_v2(X_train, y_train, lambda_reg= 0.01, epochs= 200, verbose= False, tolerance= -1)
    print(f'training time - pegasos_v2: {time.time() - begin}')
    '''
    10. Write a function classification error that takes a sparse weight vector w, 
    a list of sparse vectors X and the corresponding list of labels y, 
    and returns the fraction of errors when predicting yi using sign(wTxi). 
    In other words, the function reports the 0-1 loss of the linear predictor f(x) = wTx.
    '''
    print('--------------------------------------------------------------------------------------------')
    print('A10')
    print(f"validation set classification error using W_pegasos_v2: {classification_error(X_test, y_test, W_pegasos_v2)}")
    

    '''
    11. Search for the regularization parameter that gives the minimal percent error on your test set. 
    You should now use your faster Pegasos implementation, and run it to convergence. 
    A good search strategy is to start with a set of regularization parameters spanning a broad range of orders of magnitude. 
    Then, continue to zoom in until you're convinced that additional search will not significantly improve your test performance. 
    Plot the test errors you obtained as a function of the parameters λ you tested. 
    (Hint: the error you get with the best regularization should be closer to 15% than 20%. If not, maybe you did not train to convergence.)
    '''
    print('--------------------------------------------------------------------------------------------')
    print('A11')
    plot_lambda_vs_error(X_train, y_train, X_test, y_test, [1e-10, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], epochs=100)
    
    '''
    12. Break the predictions on the test set into groups based on the score 
    (you can play with the size of the groups to get a result you think is informative). 
    For each group, examine the percentage error. You can make a table or graph. 
    Summarize the results. Is there a correlation between higher magnitude scores and accuracy?
    '''
    print('--------------------------------------------------------------------------------------------')
    print('A12')
    W_pegasos_v2 = pegasos_v2(X_train, y_train, lambda_reg= 1e-4, epochs= 50, verbose= False, tolerance= 1e-3)
    plot_score_grps_vs_percentage_error(X_test, y_test, W_pegasos_v2, bins= 8)
    '''
    A: Higher scores -> lesser classification error -> higher accuracy
    '''
    
__main__()