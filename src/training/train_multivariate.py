import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_utils.data_loader import DataLoader
from src.models.multivariate import MultiVarPredictor

def train_evaluate_visualize_multi_var_scaled(file_paths: list, language: str = 'English'):
    all_r2_train, all_mse_train, all_mae_train = [], [], []
    all_r2_test, all_mse_test, all_mae_test = [], [], []

    for i in range(len(file_paths)):
        test_files = [file_paths[i]]
        train_files = file_paths[:i] + file_paths[i+1:]
        
        predictor = MultiVarPredictor()
        train_loader = DataLoader(train_files)
        columns_to_load = ["content", "tiktoken_r50k_base_len"]
        if language != 'all':
            columns_to_load.append("language")
        train_data_generator = train_loader.load_raw_data(columns=columns_to_load)
        for chunk in tqdm(train_data_generator, desc=f"Fold {i+1} Training"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()
            if not target_chunk.empty:
                char_counts = target_chunk['content'].astype(str).str.len()
                word_counts = target_chunk['content'].astype(str).str.split().str.len()
                x_chunk = list(zip(char_counts.tolist(), word_counts.tolist()))
                y_chunk = target_chunk["tiktoken_r50k_base_len"].tolist()
                if x_chunk:
                    predictor.partial_fit(x_chunk, y_chunk)

        y_true_train_all, y_pred_train_all = [], []
        scaled_points_for_plot = []

        train_eval_loader = DataLoader(train_files)
        train_eval_generator = train_eval_loader.load_raw_data(columns=columns_to_load)

        for train_chunk in tqdm(train_eval_generator, desc=f"Fold {i+1} In-Sample Eval"):
            target_chunk = train_chunk
            if language != 'all':
                target_chunk = train_chunk[train_chunk['language'] == language].copy()
            if not target_chunk.empty:
                char_counts = target_chunk['content'].astype(str).str.len()
                word_counts = target_chunk['content'].astype(str).str.split().str.len()
                token_counts = target_chunk["tiktoken_r50k_base_len"]
                
                y_true_train_all.extend(token_counts.tolist())
                y_pred_chunk = [predictor.predict([c, w]) for c, w in zip(char_counts, word_counts)]
                y_pred_train_all.extend(y_pred_chunk)
                
                x_raw_chunk = np.array(list(zip(char_counts, word_counts)))
                y_raw_chunk = np.array(token_counts).reshape(-1, 1)
                
                if x_raw_chunk.shape[0] > 0 and predictor.is_fitted:
                    x_scaled_chunk = predictor.x_scaler.transform(x_raw_chunk)
                    y_scaled_chunk = predictor.y_scaler.transform(y_raw_chunk)
                    scaled_points_for_plot.extend(np.hstack((x_scaled_chunk, y_scaled_chunk)).tolist())

        if y_true_train_all:
            all_r2_train.append(r2_score(y_true_train_all, y_pred_train_all))
            all_mse_train.append(mean_squared_error(y_true_train_all, y_pred_train_all))
            all_mae_train.append(mean_absolute_error(y_true_train_all, y_pred_train_all))

        y_true_test_all, y_pred_test_all = [], []
        test_loader = DataLoader(test_files)
        test_data_generator = test_loader.load_raw_data(columns=columns_to_load)
        for test_chunk in tqdm(test_data_generator, desc=f"Fold {i+1} Out of Sample Eval"):
            target_chunk = test_chunk
            if language != 'all':
                target_chunk = test_chunk[test_chunk['language'] == language].copy()

            if not target_chunk.empty:
                char_counts = target_chunk['content'].astype(str).str.len()
                word_counts = target_chunk['content'].astype(str).str.split().str.len()
                token_counts = target_chunk["tiktoken_r50k_base_len"]
                y_true_test_all.extend(token_counts.tolist())
                y_pred_chunk = [predictor.predict([c, w]) for c, w in zip(char_counts, word_counts)]
                y_pred_test_all.extend(y_pred_chunk)
                x_raw_chunk = np.array(list(zip(char_counts, word_counts)))
                y_raw_chunk = np.array(token_counts).reshape(-1, 1)
                if x_raw_chunk.shape[0] > 0 and predictor.is_fitted:
                    x_scaled_chunk = predictor.x_scaler.transform(x_raw_chunk)
                    y_scaled_chunk = predictor.y_scaler.transform(y_raw_chunk)
                    scaled_points_for_plot.extend(np.hstack((x_scaled_chunk, y_scaled_chunk)).tolist())

        if y_true_test_all:
            all_r2_test.append(r2_score(y_true_test_all, y_pred_test_all))
            all_mse_test.append(mean_squared_error(y_true_test_all, y_pred_test_all))
            all_mae_test.append(mean_absolute_error(y_true_test_all, y_pred_test_all))

        if scaled_points_for_plot:
            x_scaled_plot, y_scaled_plot, z_scaled_plot = zip(*scaled_points_for_plot)
            PLOT_SCALE = 100.0
            x_plot = np.array(x_scaled_plot) / PLOT_SCALE
            y_plot = np.array(y_scaled_plot) / PLOT_SCALE
            z_plot = np.array(z_scaled_plot) / PLOT_SCALE

            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_plot, y_plot, z_plot, c='blue', marker='o', s=5, alpha=0.05, label='Data Points')
            scaled_coeffs = predictor.model.coef_
            scaled_intercept = predictor.model.intercept_[0]

            x_surf_range = np.linspace(min(x_plot), max(x_plot), 20)
            y_surf_range = np.linspace(min(y_plot), max(y_plot), 20)
            x_surf, y_surf = np.meshgrid(x_surf_range, y_surf_range)
            z_surf = (scaled_coeffs[0] * (x_surf * PLOT_SCALE) + scaled_coeffs[1] * (y_surf * PLOT_SCALE) + scaled_intercept) / PLOT_SCALE

            ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Fitted Plane')
            ax.set_title(f'3D Scaled Space Analysis (Fold {i+1} - Raw Data)', fontsize=16)
            ax.set_xlabel(f'Scaled Char Count (/{PLOT_SCALE})', fontsize=12)
            ax.set_ylabel(f'Scaled Word Count (/{PLOT_SCALE})', fontsize=12)
            ax.set_zlabel(f'Scaled Token Count (/{PLOT_SCALE})', fontsize=12)
            filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
            plot_filename = f"results/multivariate_regression/plots/raw_3d_fit_scaled_{filename_scope}_fold_{i+1}.png"
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
            plt.close()

        model_filename = f"saved_models/multivariate_regression/raw_multi_var_model_{filename_scope}_fold_{i+1}.joblib"
        predictor.save(model_filename)


    print("In Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_train):.4f}")
    print(f"MSE: {np.mean(all_mse_train):,.2f}")
    print(f"MAE: {np.mean(all_mae_train):,.2f}")

    print("Out of Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_test):.4f}")
    print(f"MSE: {np.mean(all_mse_test):,.2f}")
    print(f"MAE: {np.mean(all_mae_test):,.2f}")

def train_evaluate_multi_var_preprocessed_scaled(file_paths: list, language: str = 'English', iqr_multiplier: float = 1.5):
    all_r2_train, all_mse_train, all_mae_train = [], [], []
    all_r2_test, all_mse_test, all_mae_test = [], [], []

    for i in range(len(file_paths)):
        test_files = [file_paths[i]]
        train_files = file_paths[:i] + file_paths[i+1:]
        
        predictor = MultiVarPredictor()
        train_loader = DataLoader(train_files)
        train_data_generator = train_loader.load_preprocessed_data(language=language, iqr_multiplier=iqr_multiplier)

        for chunk in tqdm(train_data_generator, desc=f"Fold {i+1} Training"):
            char_counts = chunk['content'].astype(str).str.len()
            word_counts = chunk['content'].astype(str).str.split().str.len()
            x_chunk = list(zip(char_counts.tolist(), word_counts.tolist()))
            y_chunk = chunk["tiktoken_r50k_base_len"].tolist()
            if x_chunk:
                predictor.partial_fit(x_chunk, y_chunk)

        y_true_train_all, y_pred_train_all = [], []
        scaled_points_for_plot = []
        train_eval_loader = DataLoader(train_files)
        train_eval_generator = train_eval_loader.load_preprocessed_data(language=language, iqr_multiplier=iqr_multiplier)
        for train_chunk in tqdm(train_eval_generator, desc=f"Fold {i+1} In Sample Eval"):
            char_counts = train_chunk['content'].astype(str).str.len()
            word_counts = train_chunk['content'].astype(str).str.split().str.len()
            token_counts = train_chunk["tiktoken_r50k_base_len"]
            y_true_train_all.extend(token_counts.tolist())
            y_pred_chunk = [predictor.predict([c, w]) for c, w in zip(char_counts, word_counts)]
            y_pred_train_all.extend(y_pred_chunk)
            x_raw_chunk = np.array(list(zip(char_counts, word_counts)))
            y_raw_chunk = np.array(token_counts).reshape(-1, 1)
            if x_raw_chunk.shape[0] > 0 and predictor.is_fitted:
                x_scaled_chunk = predictor.x_scaler.transform(x_raw_chunk)
                y_scaled_chunk = predictor.y_scaler.transform(y_raw_chunk)
                scaled_points_for_plot.extend(np.hstack((x_scaled_chunk, y_scaled_chunk)).tolist())

        if y_true_train_all:
            all_r2_train.append(r2_score(y_true_train_all, y_pred_train_all))
            all_mse_train.append(mean_squared_error(y_true_train_all, y_pred_train_all))
            all_mae_train.append(mean_absolute_error(y_true_train_all, y_pred_train_all))

        y_true_test_all, y_pred_test_all = [], []
        test_loader = DataLoader(test_files)
        test_data_generator = test_loader.load_preprocessed_data(language=language, iqr_multiplier=iqr_multiplier)
        for test_chunk in tqdm(test_data_generator, desc=f"Fold {i+1} Out of Sample Eval"):
            char_counts = test_chunk['content'].astype(str).str.len()
            word_counts = test_chunk['content'].astype(str).str.split().str.len()
            token_counts = test_chunk["tiktoken_r50k_base_len"]
            y_true_test_all.extend(token_counts.tolist())
            y_pred_chunk = [predictor.predict([c, w]) for c, w in zip(char_counts, word_counts)]
            y_pred_test_all.extend(y_pred_chunk)
            x_raw_chunk = np.array(list(zip(char_counts, word_counts)))
            y_raw_chunk = np.array(token_counts).reshape(-1, 1)
            if x_raw_chunk.shape[0] > 0 and predictor.is_fitted:
                x_scaled_chunk = predictor.x_scaler.transform(x_raw_chunk)
                y_scaled_chunk = predictor.y_scaler.transform(y_raw_chunk)
                scaled_points_for_plot.extend(np.hstack((x_scaled_chunk, y_scaled_chunk)).tolist())
        
        if y_true_test_all:
            all_r2_test.append(r2_score(y_true_test_all, y_pred_test_all))
            all_mse_test.append(mean_squared_error(y_true_test_all, y_pred_test_all))
            all_mae_test.append(mean_absolute_error(y_true_test_all, y_pred_test_all))

        if scaled_points_for_plot:
            x_scaled_plot, y_scaled_plot, z_scaled_plot = zip(*scaled_points_for_plot)
            PLOT_SCALE = 100.0
            x_plot = np.array(x_scaled_plot) / PLOT_SCALE
            y_plot = np.array(y_scaled_plot) / PLOT_SCALE
            z_plot = np.array(z_scaled_plot) / PLOT_SCALE

            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_plot, y_plot, z_plot, c='blue', marker='o', s=5, alpha=0.05, label='Data Points')
            scaled_coeffs = predictor.model.coef_
            scaled_intercept = predictor.model.intercept_[0]
            
            x_surf_range = np.linspace(min(x_plot), max(x_plot), 20)
            y_surf_range = np.linspace(min(y_plot), max(y_plot), 20)
            x_surf, y_surf = np.meshgrid(x_surf_range, y_surf_range)
            z_surf = (scaled_coeffs[0] * (x_surf * PLOT_SCALE) + scaled_coeffs[1] * (y_surf * PLOT_SCALE) + scaled_intercept) / PLOT_SCALE
            
            ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Fitted Plane')
            ax.set_title(f'3D Scaled Space Analysis (Fold {i+1} - Preprocessed)', fontsize=16)
            ax.set_xlabel(f'Scaled Char Count (/{PLOT_SCALE})', fontsize=12)
            ax.set_ylabel(f'Scaled Word Count (/{PLOT_SCALE})', fontsize=12)
            ax.set_zlabel(f'Scaled Token Count (/{PLOT_SCALE})', fontsize=12)
            filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
            plot_filename = f"results/multivariate_regression/plots/preprocessed_3d_fit_scaled_{filename_scope}_fold_{i+1}.png"
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
            plt.close()

        model_filename = f"saved_models/multivariate_regression/preprocessed_multi_var_model_{filename_scope}_fold_{i+1}.joblib"
        predictor.save(model_filename)

    print("In Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_train):.4f}")
    print(f"MSE: {np.mean(all_mse_train):,.2f}")
    print(f"MAE: {np.mean(all_mae_train):,.2f}")
    
    print("Out of Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_test):.4f}")
    print(f"MSE: {np.mean(all_mse_test):,.2f}")
    print(f"MAE: {np.mean(all_mae_test):,.2f}")