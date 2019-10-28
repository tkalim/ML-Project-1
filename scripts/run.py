import numpy as np
from scripts.implementations import *
from scripts.proj1_helpers import *

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"
RATIO = 0.8
SEED = 1
DEGREES = []
LAMBDAS = []


def load_and_prep(data_path):
    """loads the data and preps it by splitting it 
    into buckets and normalizing it
    """
    y, X, ids = load_csv_data(data_path)
    buckets = get_buckets(X)
    y_buckets = []
    for i in range(len(buckets)):
        y_buckets.append(buckets[i][:, -1], 1)
        buckets[i] = np.delete(y_buckets[i], -1, 1)
        # Normalize the data
        buckets[i] = normalize(buckets[i])
        # Add Intercept
        buckets[i] = np.column((np.ones((buckets[i].shape[0], 1)), buckets[i]))
    return X, buckets, y_buckets, ids


def get_accuracy(train_buckets, test_buckets, test_y_buckets, weights, degree_and_lambda):
    """Calculates the accuracy by comparing the predictions with given test data.

    Parameters:
    weights: The optimal weights after the training of each bucket
    test_buckets: The x on which we wish to proceed prediction, divided into buckets
    test_y_buckets: the actual output with which the comparison is going to take place, divided into buckets

    Returns: 
    the accuracy score
    """
    correct_predictions = 0
    len_data = 0
    for i in range(len(train_buckets)):
        degree = degree_and_lambda[i]["degree"]
        rr_accuracy = compute_accuracy(
            weights[i], build_poly(test_buckets[i], degree), test_y_buckets[i]
        )
        correct_predictions += rr_accuracy * len(build_poly(test_buckets[i], degree))
        len_data += len(build_poly(test_buckets[i], degree))
    total_accuracy = correct_predictions / len_data
    print(f"Accuracy = {total_accuracy}")
    return total_accuracy


def train(buckets, y_buckets, degrees, lambdas):
    """trains the models and outputs the weights for each model
    """
    print("start training")
    weights = []
    for i in range(len(buckets)):
        degree = best_degree_lambda[i]["degree"]  # TODO: change it
        lambda_ = best_degree_lambda[i]["lambda"]  # TODO: change it with correct
        w_rr, loss_rr = ridge_regression(
            y=y_buckets[i], tx=build_poly(buckets[i], degree), lambda_=lambda_
        )
        print(w_rr.shape)
        weights.append(w_rr)
        print(f"trained on bucket {i}")
    return weights


def predict(ids, x, buckets, weights, degrees=DEGREES):
    """Predicts labels according to previously trained weights

    Parameters:
    weights: The optimal weights after the training of each bucket
    x: The x on which we wish to proceed prediction
    buckets: x split into buckets
    degrees: Degree with which we expand the features
    ids: ids of submission data

    Returns: 
    submission and ids of submission
    """
    ids_array = ids
    pri_jet_num_feature = x[:, 22]
    der_mass_mmc_col_feature = x[:, 0]
    ids_array = np.column_stack((ids_array, pri_jet_num_feature))
    ids_array = np.column_stack((ids_array, der_mass_mmc_col_feature))
    id_buckets = get_id_buckets(ids_array)
    prediction = predict_labels(weights[0], build_poly(buckets[0], degrees[0]))
    prediction = np.column_stack((prediction, id_buckets[0]))
    for i in range(1, len(weights)):
        degree = DEGREES[i]
        predictions = predict_labels(weights[i], build_poly(buckets[i], degree))
        predictions = np.column_stack((predictions, id_buckets[i]))
        prediction = np.concatenate((prediction, predictions))
    prediction = prediction[prediction[:, 1].argsort()]
    prediction = prediction[:, 0]

    return prediction, ids


if __name__ == "__main__":
    train_x, buckets, y_buckets, ids_train = load_and_prep(TRAIN_DATA_PATH)
    submission_x, submission_buckets, submission_y_buckets, ids_submission = load_and_prep(
        TEST_DATA_PATH
    )

    weights = train(
        train_buckets=buckets, y_buckets=y_buckets, degrees=DEGREES, lambdas=LAMBDAS
    )

    submission, ids_submission = predict(ids=ids_submission, x=submission_x, buckets=submission_buckets, weights=weights, degrees=DEGREES)

    create_csv_submission(ids_submission, submission, "../data/output.csv")

