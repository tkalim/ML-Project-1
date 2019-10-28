import numpy as np
from implementations import *
from proj1_helpers import *

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"
OUTPUT_FILEPATH = "../data/output2.csv"
RATIO = 0.8
SEED = 1
DEGREES_LAMBDAS = [
    {"degree": 9, "lambda": 0.31461538461538463},
    {"degree": 9, "lambda": 0.44153846153846155},
    {"degree": 9, "lambda": 0.7461538461538462},
    {"degree": 9, "lambda": 0.796923076923077},
    {"degree": 9, "lambda": 0.8984615384615384},
    {"degree": 9, "lambda": 0.4923076923076923},
    {"degree": 9, "lambda": 1.0},
    {"degree": 9, "lambda": 0.796923076923077},
]


def load_and_prep(data_path):
    """loads the data and preps it by splitting it 
    into buckets and normalizing it
    """
    print(f"loading {data_path}")
    y, X, ids = load_csv_data(data_path, sub_sample=False)
    X = np.column_stack((X, y))
    buckets = get_buckets(X)
    y_buckets = []
    for i in range(len(buckets)):
        y_buckets.append(buckets[i][:, -1])
        buckets[i] = np.delete(buckets[i], -1, 1)
        # Normalize the data
        buckets[i] = normalize(buckets[i])
        # Add Intercept
        buckets[i] = np.column_stack((np.ones((buckets[i].shape[0], 1)), buckets[i]))
    print(f"{data_path} loaded and prepped")
    return X, buckets, y_buckets, ids


def get_accuracy(
    train_buckets, test_buckets, test_y_buckets, weights, degree_and_lambda
):
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


def train(buckets, y_buckets, degrees_lambdas):
    """trains the models and outputs the weights for each model
    """
    print("start training")
    weights = []
    for i in range(len(buckets)):
        degree = degrees_lambdas[i]["degree"]  # TODO: change it
        lambda_ = degrees_lambdas[i]["lambda"]  # TODO: change it with correct
        w_rr, loss_rr = ridge_regression(
            y=y_buckets[i], tx=build_poly(buckets[i], degree), lambda_=lambda_
        )
        #print(w_rr.shape)
        weights.append(w_rr)
        print(f"trained on bucket {i}")
    print("training complete")
    return weights


def predict(ids, x, buckets, weights, degrees_lambdas=DEGREES_LAMBDAS):
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
    prediction = predict_labels(
        weights[0], build_poly(buckets[0], degrees_lambdas[0]["degree"])
    )
    prediction = np.column_stack((prediction, id_buckets[0]))
    for i in range(1, len(weights)):
        degree = degrees_lambdas[i]["degree"]
        predictions = predict_labels(weights[i], build_poly(buckets[i], degree))
        predictions = np.column_stack((predictions, id_buckets[i]))
        prediction = np.concatenate((prediction, predictions))
    prediction = prediction[prediction[:, 1].argsort()]
    prediction = prediction[:, 0]

    print("predictions complete")
    return prediction


if __name__ == "__main__":
    train_x, buckets, y_buckets, ids_train = load_and_prep(TRAIN_DATA_PATH)

    submission_x, submission_buckets, submission_y_buckets, ids_submission = load_and_prep(
        TEST_DATA_PATH
    )

    weights = train(
        buckets=buckets, y_buckets=y_buckets, degrees_lambdas=DEGREES_LAMBDAS
    )

    submission = predict(
        ids=ids_submission,
        x=submission_x,
        buckets=submission_buckets,
        weights=weights,
        degrees_lambdas=DEGREES_LAMBDAS,
    )

    create_csv_submission(ids_submission, submission, OUTPUT_FILEPATH)
    print(f"{OUTPUT_FILEPATH} saved")