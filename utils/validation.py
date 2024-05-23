import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

import itertools

from utils.data import (
    NumericalScaling,
    get_numerical_and_categorical_indexes,
)


class CrossValidation:
    def __init__(
        self,
        df_X: pd.DataFrame,
        df_y: pd.DataFrame,
        numerical_scale: bool,
        numerical_features: list = None,
    ) -> None:
        """ """
        self.X = df_X.values
        self.y = df_y.values

        self.n_features = df_X.shape[1]
        self.numerical_scale = numerical_scale

        if numerical_scale and numerical_features is not None:
            numerical_indexes, categorical_indexes = (
                get_numerical_and_categorical_indexes(df_X, numerical_features)
            )

            self.numerical_scaling = NumericalScaling(
                numerical_indexes, categorical_indexes
            )

    def compute_numerical_scaling(
        self, X_train: np.array, X_validation: np.array
    ) -> tuple[np.array, np.array]:
        """ """

        X_train = self.numerical_scaling.run(X_train, use_saved_transformer=False)

        X_validation = self.numerical_scaling.run(
            X_validation, use_saved_transformer=True
        )

        return X_train, X_validation

    def one_split(
        self, train_index: np.array, validation_index: np.array
    ) -> tuple[float, float]:
        """ """

        # split in train and validation subsets
        X_train, X_validation = self.X[train_index], self.X[validation_index]
        y_train, y_validation = self.y[train_index], self.y[validation_index]

        if self.numerical_scale:
            X_train, X_validation = self.compute_numerical_scaling(
                X_train, X_validation
            )

        # fit model
        self.model.fit(X_train, y_train)

        # predict label
        y_pred, y_pred_proba = self.model.predict(X_validation)

        # predict evaluation metrics
        acc = accuracy_score(y_validation, y_pred)
        ll = log_loss(y_validation, y_pred_proba)

        return acc, ll

    def run(
        self,
        model: callable,
        split_method: str,
        shuffle: bool = False,
        random_state: float = None,
        n_splits: int = 5,
        test_size: float = 0.3,
        **kwargs,
    ):
        """
        Execute the cross-validation process with the specified model and splitting strategy.

        Parameters:
        - model (callable): The machine learning model to train and evaluate.
        - split_method (str): The type of cross-validation split ('standard', 'kfold', 'stratified_kfold').
        - shuffle (bool): Whether to shuffle data before splitting.
        - random_state (int, optional): Seed for random number generator.
        - n_splits (int): Number of folds for k-fold strategies.
        - test_size (float): Proportion of the dataset to include in the test split (for 'standard' strategy).
        - kwargs: Additional keyword arguments to pass to the model training method.

        Returns:
        - dict: Dictionary containing accuracy and log loss history and their averages.
        """

        self.model = model
        self.kwargs = kwargs

        if split_method == "standard":
            indexes = np.arange(self.X.shape[0])
            splits = [
                train_test_split(
                    indexes, test_size=test_size, random_state=random_state
                )
            ]
        elif split_method == "kfold":
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            splits = kf.split(self.X, self.y)
        elif split_method == "stratified_kfold":
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
            splits = skf.split(self.X, self.y)
        else:
            raise ValueError(f"Unsupported split method: {split_method}")

        self.history = {"accuracy": [], "log_loss": []}
        for train_index, validation_index in splits:
            acc, ll = self.one_split(train_index, validation_index)

            self.history["accuracy"].append(acc)
            self.history["log_loss"].append(ll)

        self.history["avg_accuracy"] = np.mean(self.history["accuracy"])
        self.history["avg_log_loss"] = np.mean(self.history["log_loss"])

        return self.history


def grid_search_cv(
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    model: object,
    grid_search_parameters: dict,
    cross_validation_split_method: str,
    numerical_scaling: bool,
    numerical_features: list,
    shuffle: bool,
    random_state: float,
    verbose: bool = True,
) -> tuple[dict, float, dict, list]:
    # Create a list of keys and a list of lists of values
    keys = list(grid_search_parameters.keys())
    values = list(grid_search_parameters.values())

    # Generate all combinations of the parameter values
    combinations = itertools.product(*values)

    # initiate list to store each combination of parameters tested
    parameters = []

    # initiate list to store evaluation metrics for each combination
    history = []

    # initiate variable to save index for the optimal set of parameters
    optimal_index = -1

    # initiate variable to save the lowest seen log loss
    min_log_loss = 1000

    for i, c in enumerate(combinations):
        kwargs = dict(zip(keys, c))

        model_ = model(**kwargs)

        cv = CrossValidation(
            df_X,
            df_y["Status"],
            numerical_scale=numerical_scaling,
            numerical_features=numerical_features,
        )

        h = cv.run(
            model_,
            cross_validation_split_method,
            shuffle=shuffle,
            random_state=random_state,
            **kwargs,
        )

        parameters.append(kwargs)
        history.append(h)

        # check if current combination > optimal one
        if history[i]["avg_log_loss"] < min_log_loss:
            min_log_loss = history[i]["avg_log_loss"]
            optimal_index = i

        if verbose:
            print(kwargs)

            print(
                "[Validation Set] Average Accuracy: %.2f" % (h["avg_accuracy"] * 100),
                "%",
            )
            print("[Validation Set] Average Log-loss: %.2f \n" % h["avg_log_loss"])

    return (
        parameters[optimal_index],
        history[optimal_index]["avg_log_loss"],
        parameters,
        history,
    )


# Evaluation metrics
# ------------------
def logloss(y_true_proba: np.array, y_pred_proba: np.array) -> float:
    """
    # N: number of rows in the set
    # M: number of outcomes (i.e., 3)
    # log: natural logarithm
    # y_ij: is 1 if row i has the ground truth label j and 0 otherwise
    # p_ij: is the predicted probability that observation i belongs to class j
    """

    N, M = np.shape(y_true_proba)

    sum = 0
    for i in range(N):
        for j in range(M):
            y_ij = 1 if y_true_proba[i, j] == max(y_true_proba[i, :]) else 0
            p_ij = y_pred_proba[i, j]
            sum += y_ij * np.log(p_ij)

            # print(y_ij, y_true_proba[i, j], max(y_true_proba[i, :]))

    return -(1 / N) * sum
