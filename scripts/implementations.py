from proj1_helpers import *
import numpy as np
import matplotlib.pyplot as plt

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Parameters:
    y: the output of the data
    tx: the input of the data
    initial_w: the initial weight form which we wish to proceed the least squares GD algorithm
    max_iters: maximum iterations of the algorithm before returning the result
    gamma: the rate of the descent of the algorithm
    
    returns:
    the optimal weights and its corresponding loss
    """

    w = initial_w
    for n_iter in range(max_iters):
        grad, loss = compute_gradient(y, tx, w)
        w = w - (gamma * grad)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

    Parameters:
    y: the output of the data
    tx: the input of the data
    initial_w: the initial weight form which we wish to proceed the least squares SGD algorithm
    max_iters: maximum iterations of the algorithm before returning the result
    gamma: the rate of the descent of the algorithm
    Returns: 
    the optimal weights and its corresponding loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, _ = compute_stochastic_gradient(y_batch, tx_batch, w)
            w = w - (gamma * grad)
            loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Parameters:
    y: the output of the data
    tx: the input of the data
    Returns: 
    optimal weights and loss using the normal equations
    """
    gram = np.dot(np.transpose(tx), tx)
    gram = np.linalg.inv(gram)

    w = np.dot(gram, np.transpose(tx))
    w = np.dot(w, y)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Parameters:
    y: the input of the data
    tx: the output of the data
    lambda_: the penalizing parameter for the ridge regression

    Returns: 
    optimal weights and loss for the ridge regression
    """
    N = tx.shape[1]
    a = np.dot(np.transpose(tx), tx) + (lambda_ * np.identity(N))
    b = np.dot(np.transpose(tx), y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Parameters
    y: the output data
    tx: the input data
    initial_w: the desired initial weight to begin the algorithm of logistic regression with
    max_iters: the maximum of iterations during the algorithm
    gamma: the rate of descent of the gradient

    Returns: 
    optimal weights and its corresponding loss for logistic regression
    """
    w = initial_w
    for n_iter in range(max_iters):
        yx = np.dot(y, tx)
        yxw = np.dot(yx, w)
        log = np.log(1 + np.exp(np.dot(tx, w)))
        loss = (log - yxw).sum()

        # Update rule
        sig = sigma(np.dot(tx, w))
        sig = sig - y
        grad = np.dot(np.transpose(tx), sig)
        w = w - (gamma * grad)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Parameters:
    y: the output data
    tx: the input data
    lambda_: the parameter used for penalization for the logistic regression
    initial_w: the desired initial weight to begin the algorithm of logistic regression with
    max_iters: the maximum of iterations during the algorithm
    gamma: the rate of descent of the gradient

    Returns: 
    the optimal weights and its corresponding loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        yx = np.dot(y, tx)
        yxw = np.dot(yx, w)
        log = np.log(1 + np.exp(np.dot(tx, w)))

        # Add the 'penalty' term
        loss = (log - yxw).sum() - (lambda_ / 2) * np.square((np.linalg.norm(w)))

        # Update rule
        sig = sigma(np.dot(tx, w))
        sig = sig - y
        grad = np.dot(np.transpose(tx), sig) + (2 * lambda_ * w)
        w = w - (gamma * grad)
    return w, loss

def compute_error(y, tx, w):
    """Calculates the error in the current prediction.

    Parameters:
    y: the output data
    tx: the input data
    w: actual weight for computing the error

    Returns: 
    the error between output data and actual prediction
    """
    return y - np.dot(tx, w)


def compute_loss(y, tx, w):
    """Calculates the loss using MSE.

    Parameters:
    y: the output data
    tx: the input data
    w: the actual weight on which we will compute the loss
    Returns: 
    the loss
    """
    N = y.shape[0]
    e = compute_error(y, tx, w)
    factor = 1 / (2 * N)
    loss = (np.dot(np.transpose(e), e)) * factor
    return loss


def compute_gradient(y, tx, w):
    """Computes the gradient of the MSE loss function.

    Parameters:
    y: the output data
    tx: the input data
    w: the actual weight wishing to compute the gradient on

    Returns: 
    the gradient and its corresponding loss
    """
    N = y.shape[0]
    e = compute_error(y, tx, w)
    factor = -1 / N
    grad = (np.dot(np.transpose(tx), e)) * factor
    loss = compute_loss(y, tx, w)
    return grad, loss


def compute_stochastic_gradient(y, tx, w):
    """Computes the stochastic gradient from a few examples of n and their corresponding y_n labels.

    Parameters:
    y: the output data
    tx: the input data
    w: the actual weight on which we wish to compute the stochastic gradient

    Returns: 
    gradient and its corresponding loss
    """
    N = y.shape[0]
    e = compute_error(y, tx, w)
    factor = -1 / N
    grad = (np.dot(np.transpose(tx), e)) * factor
    loss = compute_loss(y, tx, w)
    return grad, loss


def sigma(x):
    """Calculates sigma using the given formula.

    Parameters:
    x: a given vector

    Returns: 
    sigma function applied on the given vector
    """
    return np.exp(x) / (1 + np.exp(x))


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generates a minibatch iterator for a dataset.
    Takes as input two iterables - the output desired values 'y' and the input data 'tx'.
    Outputs an iterator which gives mini-batches of batch_size matching elements from y and tx.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use:
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        do something
    
    Parameters:
    y: output data
    tx: input data
    batch_size: the size of batches on which we will compute gradient descent
    num_batches: the number of batches
    shuffle: boolean parameter being true in orther to randomly shuffle the data

    Returns: 
    shuffled batches on which we can proceed SGD
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_accuracy(w_train, x, y):
    """Calculates the accuracy by comparing the predictions with given test data.

    Parameters:
    w_train: The optimal weights after the training
    x: The x on which we wish to proceed prediction
    y: the actual output with which the comparaison is going to take place

    Returns: 
    the accuracy score
    """
    pred = predict_labels(w_train, x)
    N = len(pred)
    count = 0.0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            count += 1
    return count/N


def split_data(tx, ty, ratio, seed=1):
    """Split the training data by ratio.

    Parameters:
    tx: The input data
    ty: the output data
    ratio: the ratio used for the split
    seed: the seed generating randomness

    Returns: 
    the training and testing sets with its inputs and outputs
    """
    np.random.seed(seed)
    split_idxs = [i for i in range(len(tx))]

    # Shuffle the indicies randomly
    np.random.shuffle(split_idxs)
    tx_shuffled = tx[split_idxs]
    ty_shuffled = ty[split_idxs]

    # Split by ratio
    split_pos = int(len(tx) * ratio)
    x_train = tx_shuffled[:split_pos]
    x_test = tx_shuffled[split_pos:]
    y_train = ty_shuffled[:split_pos]
    y_test = ty_shuffled[split_pos:]

    return x_train, y_train, x_test, y_test


def build_poly(x, degree):
    """Builds a polynomial of the given degree and appends it to the given matrix.

    Parameters:
    x: The matrix on which we wish to build the polynomial degree
    degree: the polynomial degree built

    Returns: 
    a matrix having the polynomial degree built
    """
    x_ret = x
    for i in range(2, degree+1):
        x_ret = np.c_[x_ret, np.power(x, i)]
    return x_ret


def normalize(x):
    """Standardizes the matrix by subtracting mean of each column and then dividing by standard deviation.

    Parameters:
    x: The data we want to standardize

    Returns: 
    the data standardized
    """
    res = x.copy()
    for i in range(0, res.shape[1]):
        if i == 22:
            continue

        # Calculate mean and standard deviation without including NaN values
        mean = np.nanmean(res[:, i])
        std = np.nanstd(res[:, i])
        median = np.nanmedian(res[:, i])


        # Change mean and standard deviation if column has all NaN values
        if np.isnan(median):
            median = 0
        if np.isnan(std) or std == 0:
            std = 1

        # Replaces NaN values with mean and divides by standard deviation
        for j in range(len(res[:, i])):
            if np.isnan(res[j][i]):
                res[j][i] = median
                # print(median)
                res[j][i] -= mean
            else:
                res[j][i] -= mean
            res[j][i] /= std
    return res


def replace_999_with_nan(x):
    """Replaces -999 (undefined values) with NaN.

    Parameters:
    x: data with 999 values

    Returns: 
    data having "999" replaced with "nan"
    """
    res = x.copy()
    res[res == -999.0] = np.nan
    return res


def get_buckets(x):
    """Splits the dataset into 8 buckets.
    Based on 4 values (0, 1, 2, 3) of PRI_jet_num and 2 values (defined or -999) of DER_mass_MMC.

    Parameters:
    x: Our dataset

    Returns: 
    different buckets from the dataset based on some categorical values
    """
    result = []
    for i in range(0, 4):
        # Get all rows where PRI_jet_num equals i
        x_jet_feature = x[x[:, 22] == i]

        # Get all rows where DER_mass_MMC defined and undefined
        xi_defined = x_jet_feature[x_jet_feature[:, 0] != -999.0]
        xi_undefined = x_jet_feature[x_jet_feature[:, 0] == -999.0]

        result.append(xi_defined)
        result.append(xi_undefined)
    return result


def get_id_buckets(x):
    """Splits the set of ids into 8 buckets as above.
    Used for sorting the predictions based on event id.

    Parameters:
    x: Our dataset

    Returns: 
    getting the ids of the buckets generated with the methode get_buckets
    """
    id_buckets = []
    for i in range(0, 4):
        x_jet_feature = x[x[:, 1] == i]
        xi_defined = x_jet_feature[x_jet_feature[:, -1] != -999.0]
        xi_undefined = x_jet_feature[x_jet_feature[:, -1] == -999.0]
        id_buckets.append(xi_defined)
        id_buckets.append(xi_undefined)

    return id_buckets


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Parameters:
    y: Our ouput y
    k_fold: the number of folds for the cross validation procedure
    seed: the random parameter

    Returns: 
    array of indices shuffled for the purpose of k-fold cross validation
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression.

    Parameters:
    y: output data
    x: input data
    k_indices: indices used for cross validation
    k: the kth set is the test cross validation set
    lambda_: penalising lambda of ridge regression
    degree: polynomial degree of feature engineering

    Returns: the training/testing loss considering the kth set as the test set
    """
    # get k'th subgroup in test, others in train
    test_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indices = train_indices.reshape(-1)
    y_te = y[test_indices]
    y_tr = y[train_indices]
    x_te = x[test_indices]
    x_tr = x[train_indices]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression
    w, _ = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, w))
    return loss_tr, loss_te, w


def best_degree_selection(y, x, degrees, k_fold, lambdas, seed=1):
    """

    Parameters:
    y: output data
    x: input data
    degrees: different polynomial degrees on which we wish to get the best one
    k_fold: number of folds of the cross validation procedure
    lambdas: different penalizing lambdas on which we wish to get the best
    seed: random factor

    Returns: 
    the best polynomial degree based on least mse of the best lambda
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    # vary degree
    for degree in degrees:
        # cross validation
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            for k in range(k_fold):
                _, loss_te, _ = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_tmp))

        ind_lambda_opt = np.argmin(rmse_te)

        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])

    ind_best_degree = np.argmin(best_rmses)
    print("best lambdas : " + str(lambdas[ind_best_degree]))
    print("best degree "+str(degrees[ind_best_degree]))

    return {"degree": degrees[ind_best_degree], "lambda": lambdas[ind_best_degree]}

def cross_validation_visualization(lambds, mse_tr, mse_te,b):
    """visualization the curves of mse_tr and mse_te.

    Parameters:
    lambds: Different penalizing lambdas
    mse_tr: mean squared error of the training set
    mse_te: mean squared error of the testing set
    b: an integer referring to the bucket on which we are cross validating

    Returns: 
    cross validation figure
    """
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation for bucket "+str(b))