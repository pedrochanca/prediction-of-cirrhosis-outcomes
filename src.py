import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer


# Load Data
# ---------
def load_data(
    KAGGLE: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function loads the train and test sets.

    The files either come from Kaggle's storage unit or from the local directory.

    Parameters
    ----------
    KAGGLE (bool): if True, load from kaggle. Else, load from local directory.

    Notes
    -----
    # From Kaggle:
    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
    """

    if KAGGLE:
        df_train = pd.read_csv("/kaggle/input/playground-series-s3e26/train.csv")
        df_X_test = pd.read_csv("/kaggle/input/playground-series-s3e26/test.csv")
        df_y_test = pd.read_csv(
            "/kaggle/input/playground-series-s3e26/sample_submission.csv"
        )
    else:
        df_train = pd.read_csv("/datasets/train.csv")
        df_X_test = pd.read_csv("/datasets/test.csv")
        df_y_test = pd.read_csv("/datasets/sample_submission.csv")

    # Transform the raw data
    df_y_train = df_train["Status"]
    df_X_train = df_train.drop(["Status", "id"], axis=1)
    df_X_test = df_X_test.drop("id", axis=1)

    return df_X_train, df_X_test, df_y_train, df_y_test


# Data Transformation
# -------------------
def categorical_to_numerical(
    df: pd.DataFrame, features: list[str], transformer: str
) -> pd.DataFrame:
    if transformer == "ordinal":
        transformer = OrdinalEncoder(handle_unknown="error")
    elif transformer == "one-hot":
        transformer = OneHotEncoder(handle_unknown="error")

    # Create a column transformer to apply transformations to the appropriate columns
    preprocessor = ColumnTransformer(transformers=[("cat", transformer, features)])

    # Fit and transform the data -> The result is a NumPy array
    transformed_data = preprocessor.fit_transform(df[features])

    # Get feature names after encoding (pertinent for one-hot encoding)
    feature_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(features)
    )

    # Convert the numpy array to a DataFrame
    df_transformed = pd.DataFrame(transformed_data, columns=feature_names)

    return pd.concat([df[df.columns.difference(features)], df_transformed], axis=1)


def numerical_scaling(
    x_values: np.array, transformer: callable = None
) -> tuple[pd.DataFrame, callable]:
    if transformer is None:
        # Create transformers for numerical and categorical features
        transformer = StandardScaler()

        # Fit and transform the data -> The result is a NumPy array
        transformed_data = transformer.fit_transform(x_values)

    else:
        transformed_data = transformer.transform(x_values)

    return transformed_data, transformer


def encode_label(y_values: np.array, label_order: list[str]) -> np.array:
    label_encoder = LabelEncoder()
    label_encoder.fit(label_order)  # Fits the encoder in the order you specify

    # Transform labels
    return label_encoder.transform(y_values)


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
