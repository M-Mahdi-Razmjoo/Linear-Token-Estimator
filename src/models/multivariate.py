import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from typing import List

class MultiVarPredictor:
    def __init__(self):
        self.model = SGDRegressor(
            loss='huber',
            penalty=None,
            alpha=0.0001,
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000,
            tol=1e-4,
            warm_start=True,
            random_state=42
        )
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.is_fitted = False

    def partial_fit(self, x_data: List[List[float]], y_data: List[float]):
        '''Incrementally fits the model using new batches of X and Y data. Input and output values are scaled before training.'''
        x_np = np.array(x_data)
        y_np = np.array(y_data).reshape(-1, 1)

        self.x_scaler.partial_fit(x_np)
        self.y_scaler.partial_fit(y_np)

        x_scaled = self.x_scaler.transform(x_np)
        y_scaled = self.y_scaler.transform(y_np)
        
        self.model.partial_fit(x_scaled, y_scaled.ravel())
        self.is_fitted = True

    def predict(self, x_values: List[float]) -> int:
        '''Predicts the output for a given input vector. Input is scaled, prediction is made, and the result is inverse-transformed to the original scale.'''
        x_reshaped = np.array([x_values])
        x_scaled = self.x_scaler.transform(x_reshaped)
        prediction_scaled = self.model.predict(x_scaled).reshape(-1, 1)
        prediction_original_scale = self.y_scaler.inverse_transform(prediction_scaled)
        return int(round(prediction_original_scale[0, 0]))

    def get_model_params(self) -> dict:
        '''Returns the model's learned coefficients and intercept in the original input/output scale.'''
        scaled_coefficients = self.model.coef_
        scaled_intercept = self.model.intercept_[0]

        x_means = self.x_scaler.mean_
        x_stds = self.x_scaler.scale_
        y_mean = self.y_scaler.mean_[0]
        y_std = self.y_scaler.scale_[0]

        final_coefficients = scaled_coefficients * (y_std / x_stds)
        final_intercept = y_mean + (scaled_intercept * y_std) - np.sum(final_coefficients * x_means)

        return {
            'coefficients': final_coefficients.tolist(),
            'intercept': final_intercept
        }

    def save(self, file_path: str):
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str):
        predictor = joblib.load(file_path)
        return predictor