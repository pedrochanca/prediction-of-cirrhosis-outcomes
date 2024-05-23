import tensorflow as tf
from tensorflow import keras as kr
import numpy as np


class SoftmaxRegressionNN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.compile()

    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying model if it's not found in this class.
        If it's a callable attribute, wrap it to modify behavior.
        """
        orig_attr = getattr(self.model, attr)
        if callable(orig_attr):

            def hooked(*args):
                # Modify args or kwargs here before passing them to the original function
                # For example, let's filter out kwargs for 'fit' method specifically
                if attr == "fit":
                    allowed_kwargs = [
                        "epochs",
                        "batch_size",
                        "verbose",
                        # "callbacks",
                        # "validation_data",
                    ]
                    kwargs_ = {
                        k: v for k, v in self.kwargs.items() if k in allowed_kwargs
                    }

                # Now call the original attribute with potentially modified arguments
                return orig_attr(*args, **kwargs_)

            return hooked
        else:
            return orig_attr

    def compile(self) -> callable:
        model = self.kwargs["sequential"]

        allowed_kwargs = ["learning_rate"]
        kwargs_ = {k: v for k, v in self.kwargs.items() if k in allowed_kwargs}

        model.compile(
            loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=kr.optimizers.legacy.Adam(
                **kwargs_
            ),  # legacy.Adam - Adam runs slow on M1/M2 chips
            metrics=["sparse_categorical_crossentropy"],
        )

        return model

    def predict(self, X_values: np.array) -> tuple[np.array, np.array]:
        allowed_kwargs = [
            "verbose",
        ]
        kwargs_ = {k: v for k, v in self.kwargs.items() if k in allowed_kwargs}

        prediction = self.model.predict(X_values, **kwargs_)
        self.y_pred_proba = tf.nn.softmax(prediction).numpy()
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)

        return self.y_pred, self.y_pred_proba


# Sequential Models for validation testig
# ---------------------------------------
def build_models():
    tf.random.set_seed(20)

    model_1 = kr.models.Sequential(
        layers=[
            kr.layers.Dense(25, "relu"),
            kr.layers.Dense(10, "relu"),
            kr.layers.Dense(3, "linear"),
        ],
        name="model_1",
    )

    model_list = [model_1]

    return model_list
