import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class Predictor:
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

    def partial_fit(self, x_data: list, y_data: list):
        '''Incrementally fits the model.'''
        x_reshaped = np.array(x_data).reshape(-1, 1)
        y_reshaped = np.array(y_data).reshape(-1, 1)
        
        self.x_scaler.partial_fit(x_reshaped)
        x_scaled = self.x_scaler.transform(x_reshaped)

        self.y_scaler.partial_fit(y_reshaped)
        y_scaled = self.y_scaler.transform(y_reshaped)
        
        self.model.partial_fit(x_scaled, y_scaled.ravel()) 
        self.is_fitted = True

    def predict(self, x_value: int) -> int:
        '''Returns the model's learned coefficients.'''
        x_reshaped = np.array([[x_value]])
        x_scaled = self.x_scaler.transform(x_reshaped)
        prediction_scaled = self.model.predict(x_scaled).reshape(-1, 1)
        prediction_original_scale = self.y_scaler.inverse_transform(prediction_scaled)
        return int(round(prediction_original_scale[0, 0]))
    
    def get_model_params(self) -> dict:
        scaled_a = self.model.coef_[0]
        scaled_b = self.model.intercept_[0]

        x_mean = self.x_scaler.mean_[0]
        x_std = self.x_scaler.scale_[0]
        y_mean = self.y_scaler.mean_[0]
        y_std = self.y_scaler.scale_[0]

        # Convert parameters back to original scale
        final_a = scaled_a * (y_std / x_std)
        final_b = y_mean - final_a * x_mean + (scaled_b * y_std)
        return {'a': final_a, 'b': final_b}

    def save(self, file_path: str):
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str):
        predictor = joblib.load(file_path)
        return predictor