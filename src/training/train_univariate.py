import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_utils.data_loader import DataLoader
from src.models.univariate import Predictor

def train_and_evaluate_by_char_count_scaled(file_paths: list, language: str = 'English'):
    all_r2_train, all_mse_train, all_mae_train = [], [], []
    all_r2_test, all_mse_test, all_mae_test = [], [], []
    PLOTTING_SCALING_FACTOR = 200.0

    for i in range(len(file_paths)):        
        test_files = [file_paths[i]]
        train_files = file_paths[:i] + file_paths[i+1:]
        
        predictor = Predictor()
        train_loader = DataLoader(train_files)
        columns_to_load = ["content", "tiktoken_r50k_base_len"]
        if language != 'all':
            columns_to_load.append("language")

        train_data_generator = train_loader.load_raw_data(columns=columns_to_load)
        for chunk in tqdm(train_data_generator, desc=f"Fold {i+1} Training"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()

            x_chunk = target_chunk['content'].astype(str).str.len().tolist()
            y_chunk = target_chunk["tiktoken_r50k_base_len"].tolist()
            predictor.partial_fit(x_chunk, y_chunk)

        y_true_train_all, y_pred_train_all = [], []
        x_scaled_train, y_scaled_train = [], []
        train_eval_loader = DataLoader(train_files)
        train_eval_generator = train_eval_loader.load_raw_data(columns=columns_to_load)
        for train_chunk in tqdm(train_eval_generator, desc=f"Fold {i+1} In Sample Eval"):
            target_chunk = train_chunk
            if language != 'all':
                target_chunk = train_chunk[train_chunk['language'] == language].copy()

            x_train_chunk_raw = target_chunk['content'].astype(str).str.len().tolist()
            y_true_train_chunk = target_chunk["tiktoken_r50k_base_len"].tolist()
            y_pred_train_chunk = [predictor.predict(x) for x in x_train_chunk_raw]

            y_true_train_all.extend(y_true_train_chunk)
            y_pred_train_all.extend(y_pred_train_chunk)

            x_train_chunk_2d = np.array(x_train_chunk_raw).reshape(-1, 1)
            y_train_chunk_2d = np.array(y_true_train_chunk).reshape(-1, 1)
            x_scaled_train.extend(predictor.x_scaler.transform(x_train_chunk_2d).flatten().tolist())
            y_scaled_train.extend(predictor.y_scaler.transform(y_train_chunk_2d).flatten().tolist())

        all_r2_train.append(r2_score(y_true_train_all, y_pred_train_all))
        all_mse_train.append(mean_squared_error(y_true_train_all, y_pred_train_all))
        all_mae_train.append(mean_absolute_error(y_true_train_all, y_pred_train_all))

        y_true_test_all, y_pred_test_all = [], []
        x_scaled_test, y_scaled_test = [], []

        test_loader = DataLoader(test_files)
        test_data_generator = test_loader.load_raw_data(columns=columns_to_load)
        for test_chunk in tqdm(test_data_generator, desc=f"Fold {i+1} Out of Sample Eval"):
            target_chunk = test_chunk
            if language != 'all':
                target_chunk = test_chunk[test_chunk['language'] == language].copy()

            x_test_chunk_raw = target_chunk['content'].astype(str).str.len().tolist()
            y_test_chunk_raw = target_chunk["tiktoken_r50k_base_len"].tolist()
            y_pred_chunk = [predictor.predict(x) for x in x_test_chunk_raw]

            y_true_test_all.extend(y_test_chunk_raw)
            y_pred_test_all.extend(y_pred_chunk)

            x_test_chunk_2d = np.array(x_test_chunk_raw).reshape(-1, 1)
            y_test_chunk_2d = np.array(y_test_chunk_raw).reshape(-1, 1)
            x_scaled_test.extend(predictor.x_scaler.transform(x_test_chunk_2d).flatten().tolist())
            y_scaled_test.extend(predictor.y_scaler.transform(y_test_chunk_2d).flatten().tolist())

        all_r2_test.append(r2_score(y_true_test_all, y_pred_test_all))
        all_mse_test.append(mean_squared_error(y_true_test_all, y_pred_test_all))
        all_mae_test.append(mean_absolute_error(y_true_test_all, y_pred_test_all))

        x_plot_original = np.array(x_scaled_train + x_scaled_test)
        y_plot_original = np.array(y_scaled_train + y_scaled_test)

        x_plot = x_plot_original / PLOTTING_SCALING_FACTOR
        y_plot = y_plot_original / PLOTTING_SCALING_FACTOR
        fig, axs = plt.subplots(2, 3, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [0.5, 1, 2.5]})
        gs = axs[0, 0].get_gridspec()
        for ax in axs[0, :]: ax.remove()
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.scatter(x_plot, y_plot, alpha=0.1, s=15, label='Data Points', edgecolors='none')
        a_scaled = predictor.model.coef_[0]
        b_scaled = predictor.model.intercept_
        x_line_original = np.array([min(x_plot_original), max(x_plot_original)])
        y_line_original = a_scaled * x_line_original + b_scaled
        ax_main.plot(x_line_original / PLOTTING_SCALING_FACTOR, y_line_original / PLOTTING_SCALING_FACTOR, color='red', linewidth=2, label='Fitted Line')
        ax_main.set_title(f'Overall Scaled Space Analysis (Fold {i+1})', fontsize=16)
        ax_main.set_xlabel('Scaled Character Count (Divided by 200)', fontsize=12)
        ax_main.set_ylabel('Scaled Token Count (Divided by 200)', fontsize=12)
        ax_main.legend(fontsize=10)
        ax_main.grid(True, linestyle='--', linewidth=0.5)

        subplot_ranges = [(0, 0.5), (0.5, 1.5), (1.5, 4.0)]
        for j, (x_min, x_max) in enumerate(subplot_ranges):
            ax = axs[1, j]
            mask = (x_plot >= x_min) & (x_plot <= x_max)
            ax.scatter(x_plot[mask], y_plot[mask], alpha=0.2, s=10, edgecolors='none')
            x_sub_line = np.array([x_min, x_max])
            y_sub_line = a_scaled * x_sub_line + (b_scaled / PLOTTING_SCALING_FACTOR)
            ax.plot(x_sub_line, y_sub_line, color='red', linewidth=2)
            ax.set_title(f'Zoom: {x_min} to {x_max}', fontsize=10)
            ax.set_xlabel('Scaled Chars', fontsize=8)
            ax.set_xlim(x_min, x_max)
            y_lim_min, y_lim_max = min(y_sub_line), max(y_sub_line)
            y_padding = (y_lim_max - y_lim_min) * 0.15
            ax.set_ylim(y_lim_min - y_padding, y_lim_max + y_padding)
            ax.grid(True, linestyle='--', linewidth=0.5)
            
        axs[1,0].set_ylabel('Scaled Tokens', fontsize=8)
        plt.tight_layout(pad=3.0)
        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"results/univariate_regression/plots/scaled_space_analysis_{filename_scope}_fold_{i+1}.png"
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
        plt.close()

        model_filename = f"saved_models/univariate_regression/model_{filename_scope}_fold_{i+1}.joblib"
        predictor.save(model_filename)

    print("In Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_train):.4f}")
    print(f"MSE: {np.mean(all_mse_train):,.2f}")
    print(f"MAE: {np.mean(all_mae_train):,.2f}")
    print("Out of Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_test):.4f}")
    print(f"MSE: {np.mean(all_mse_test):,.2f}")
    print(f"MAE: {np.mean(all_mae_test):,.2f}")

def train_and_evaluate_preprocessed_by_char_count_scaled(file_paths: list, language: str = 'English', iqr_multiplier: float = 1.5):
    all_r2_train, all_mse_train, all_mae_train = [], [], []
    all_r2_test, all_mse_test, all_mae_test = [], [], []
    PLOTTING_SCALING_FACTOR = 200.0

    for i in range(len(file_paths)):        
        test_files = [file_paths[i]]
        train_files = file_paths[:i] + file_paths[i+1:]
        predictor = Predictor()
        train_loader = DataLoader(train_files)
        train_data_generator = train_loader.load_preprocessed_data(language=language, iqr_multiplier=iqr_multiplier)
        
        for chunk in tqdm(train_data_generator, desc=f"Fold {i+1} Training"):
            x_chunk = chunk['content'].astype(str).str.len().tolist()
            y_chunk = chunk["tiktoken_r50k_base_len"].tolist()
            predictor.partial_fit(x_chunk, y_chunk)

        y_true_train_all, y_pred_train_all = [], []
        x_scaled_train, y_scaled_train = [], []
        
        train_eval_loader = DataLoader(train_files)
        train_eval_generator = train_eval_loader.load_preprocessed_data(language=language, iqr_multiplier=iqr_multiplier)
        
        for train_chunk in tqdm(train_eval_generator, desc=f"Fold {i+1} In Sample Eval"):
            if not train_chunk.empty:
                x_train_chunk_raw = train_chunk['content'].astype(str).str.len().tolist()
                y_true_train_chunk = train_chunk["tiktoken_r50k_base_len"].tolist()
                y_pred_train_chunk = [predictor.predict(x) for x in x_train_chunk_raw]
                y_true_train_all.extend(y_true_train_chunk)
                y_pred_train_all.extend(y_pred_train_chunk)
                x_train_chunk_2d = np.array(x_train_chunk_raw).reshape(-1, 1)
                y_train_chunk_2d = np.array(y_true_train_chunk).reshape(-1, 1)
                x_scaled_train.extend(predictor.x_scaler.transform(x_train_chunk_2d).flatten().tolist())
                y_scaled_train.extend(predictor.y_scaler.transform(y_train_chunk_2d).flatten().tolist())

        all_r2_train.append(r2_score(y_true_train_all, y_pred_train_all))
        all_mse_train.append(mean_squared_error(y_true_train_all, y_pred_train_all))
        all_mae_train.append(mean_absolute_error(y_true_train_all, y_pred_train_all))

        y_true_test_all, y_pred_test_all = [], []
        x_scaled_test, y_scaled_test = [], []
        test_loader = DataLoader(test_files)
        test_data_generator = test_loader.load_preprocessed_data(language=language, iqr_multiplier=iqr_multiplier)
        
        for test_chunk in tqdm(test_data_generator, desc=f"Fold {i+1} Out of Sample Eval"):
            x_test_chunk_raw = test_chunk['content'].astype(str).str.len().tolist()
            y_test_chunk_raw = test_chunk["tiktoken_r50k_base_len"].tolist()
            y_pred_chunk = [predictor.predict(x) for x in x_test_chunk_raw]

            y_true_test_all.extend(y_test_chunk_raw)
            y_pred_test_all.extend(y_pred_chunk)

            x_test_chunk_2d = np.array(x_test_chunk_raw).reshape(-1, 1)
            y_test_chunk_2d = np.array(y_test_chunk_raw).reshape(-1, 1)
            x_scaled_test.extend(predictor.x_scaler.transform(x_test_chunk_2d).flatten().tolist())
            y_scaled_test.extend(predictor.y_scaler.transform(y_test_chunk_2d).flatten().tolist())

        all_r2_test.append(r2_score(y_true_test_all, y_pred_test_all))
        all_mse_test.append(mean_squared_error(y_true_test_all, y_pred_test_all))
        all_mae_test.append(mean_absolute_error(y_true_test_all, y_pred_test_all))
        
        x_plot_original = np.array(x_scaled_train + x_scaled_test)
        y_plot_original = np.array(y_scaled_train + y_scaled_test)
        x_plot = x_plot_original / PLOTTING_SCALING_FACTOR
        y_plot = y_plot_original / PLOTTING_SCALING_FACTOR

        fig, axs = plt.subplots(2, 3, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [0.5, 1, 2.5]})
        gs = axs[0, 0].get_gridspec()
        for ax in axs[0, :]: ax.remove()
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.scatter(x_plot, y_plot, alpha=0.1, s=15, label='Data Points', edgecolors='none')
        a_scaled = predictor.model.coef_[0]
        b_scaled = predictor.model.intercept_
        x_line_original = np.array([min(x_plot_original), max(x_plot_original)])
        y_line_original = a_scaled * x_line_original + b_scaled

        ax_main.plot(x_line_original / PLOTTING_SCALING_FACTOR, y_line_original / PLOTTING_SCALING_FACTOR, 
                        color='red', linewidth=2, label='Fitted Line')
        
        ax_main.set_title(f'Overall Scaled Space Analysis (Fold {i+1} - Preprocessed)', fontsize=16)
        ax_main.set_xlabel('Scaled Character Count (Divided by 200)', fontsize=12)
        ax_main.set_ylabel('Scaled Token Count (Divided by 200)', fontsize=12)
        ax_main.legend(fontsize=10)
        ax_main.grid(True, linestyle='--', linewidth=0.5)

        subplot_ranges = [(0, 0.5), (0.5, 1.5), (1.5, 4.0)]
        for j, (x_min, x_max) in enumerate(subplot_ranges):
            ax = axs[1, j]
            mask = (x_plot >= x_min) & (x_plot <= x_max)
            ax.scatter(x_plot[mask], y_plot[mask], alpha=0.2, s=10, edgecolors='none')
            x_sub_line = np.array([x_min, x_max])
            y_sub_line = a_scaled * x_sub_line + (b_scaled / PLOTTING_SCALING_FACTOR)
            ax.plot(x_sub_line, y_sub_line, color='red', linewidth=2)
            ax.set_title(f'Zoom: {x_min} to {x_max}', fontsize=10)
            ax.set_xlabel('Scaled Chars', fontsize=8)
            ax.set_xlim(x_min, x_max)
            y_lim_min, y_lim_max = min(y_sub_line), max(y_sub_line)
            y_padding = (y_lim_max - y_lim_min) * 0.15
            ax.set_ylim(y_lim_min - y_padding, y_lim_max + y_padding)
            ax.grid(True, linestyle='--', linewidth=0.5)

        axs[1,0].set_ylabel('Scaled Tokens', fontsize=8)
        plt.tight_layout(pad=3.0)
        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"results/univariate_regression/plots/preprocessed_scaled_analysis_{filename_scope}_fold_{i+1}.png"
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
        print(f"Plot for fold {i+1} saved as {plot_filename}")
        plt.close()

        model_filename = f"saved_models/univariate_regression/preprocessed_model_{filename_scope}_fold_{i+1}.joblib"
        predictor.save(model_filename)

    print("In-Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_train):.4f}")
    print(f"MSE: {np.mean(all_mse_train):,.2f}")
    print(f"MAE: {np.mean(all_mae_train):,.2f}")
    print("Out-of-Sample Error (Average)")
    print(f"R squared: {np.mean(all_r2_test):.4f}")
    print(f"MSE: {np.mean(all_mse_test):,.2f}")
    print(f"MAE: {np.mean(all_mae_test):,.2f}")