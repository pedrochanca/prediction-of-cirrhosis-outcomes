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
        df_train = pd.read_csv("./data/train.csv")
        df_X_test = pd.read_csv("./data/test.csv")
        df_y_test = pd.read_csv("./data/sample_submission.csv")

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


def encode_label(y_values: np.array, label_order: list[str]) -> np.array:
    label_encoder = LabelEncoder()
    label_encoder.fit(label_order)  # Fits the encoder in the order you specify

    # Transform labels
    return label_encoder.transform(y_values)


class NumericalScaling:
    def __init__(self, numerical_indexes: list, categorical_indexes: list):
        self.numerical_indexes = numerical_indexes
        self.categorical_indexes = categorical_indexes

    def run(self, X_values: np.array, use_saved_transformer: bool) -> np.array:
        if not use_saved_transformer:
            # create transformer
            self.transformer = StandardScaler()

            # fit the transformer and get scaled data
            data = self.transformer.fit_transform(X_values[:, self.numerical_indexes])

        else:
            # scale data using existing transformer
            data = self.transformer.transform(X_values[:, self.numerical_indexes])

        X_values_ = np.concatenate(
            (data, X_values[:, self.categorical_indexes]), axis=1
        )

        return X_values_


def get_numerical_and_categorical_indexes(
    df_X: pd.DataFrame, numerical_features: list[str]
) -> list[int]:
    """ """

    n_features = df_X.shape[1]

    numerical_indexes = [df_X.columns.get_loc(column) for column in numerical_features]

    categorical_indexes = list(set(np.arange(n_features)) - set(numerical_indexes))

    return numerical_indexes, categorical_indexes
