import argparse
import os
from src.training import train_univariate, train_multivariate
from src.analysis.analysis_plots import DataMining
from src.utils import shuffle_and_track_data, analyze_symbol_percentage_distribution

def main():
    parser = argparse.ArgumentParser(description="Token Estimator Project CLI")
    parser.add_argument('--data-dir', type=str, default='data/raw/shuffled_data', help='Directory containing the Parquet dataset files.')
    
    subparsers = parser.add_subparsers(dest='action', help='Available actions', required=True)

    # Parser for training
    parser_train = subparsers.add_parser('train', help='Train a model')
    parser_train.add_argument('model_type', choices=['univariate', 'multivariate'], help='Type of regression model to train')
    parser_train.add_argument('--preprocessed', action='store_true', help='Use preprocessed data for training')
    parser_train.add_argument('--lang', type=str, default='all', help='Language to filter data on')
    parser_train.add_argument('--iqr', type=float, default=1.5, help='IQR multiplier for preprocessing')

    # Parser for analysis
    parser_analyze = subparsers.add_parser('analyze', help='Run data analysis and generate plots')
    parser_analyze.add_argument('plot_type', choices=['lang_comp', 'punc_hist', 'symbol_dist'], help='Type of plot to generate')
    
    # Parser for data shuffling
    parser_shuffle = subparsers.add_parser('shuffle', help='Shuffle raw data files')
    parser_shuffle.add_argument('--base-path', type=str, default='data/raw', help='Base path containing the original merged parquet files.')

    args = parser.parse_args()

    os.makedirs("results/univariate_regression/plots", exist_ok=True)
    os.makedirs("results/multivariate_regression/plots", exist_ok=True)
    os.makedirs("saved_models/univariate_regression", exist_ok=True)
    os.makedirs("saved_models/multivariate_regression", exist_ok=True)

    file_names = [f"shuffled_merged_{i}.parquet" for i in range(1, 11)]
    file_paths = [os.path.join(args.data_dir, file) for file in file_names]

    if args.action == 'shuffle':
        print(f"Shuffling data from base path: {args.base_path}")
        shuffle_and_track_data(base_data_path=args.base_path, random_seed=42)
        print("Shuffling complete.")
    
    elif args.action == 'analyze':
        print(f"Running analysis for: {args.plot_type}")
        if args.plot_type == 'symbol_dist':
            analyze_symbol_percentage_distribution(file_paths)
        else:
            data_miner = DataMining(file_paths)
            if args.plot_type == 'lang_comp':
                data_miner.plot_language_comparison(languages_to_plot='all')
            elif args.plot_type == 'punc_hist':
                data_miner.plot_punctuation_histogram(language='all')
        print("Analysis complete. Check the 'results' folder.")

    elif args.action == 'train':
        if args.model_type == 'univariate':
            print("Training univariate model...")
            if args.preprocessed:
                train_univariate.train_and_evaluate_preprocessed_by_char_count_scaled(
                    file_paths=file_paths, language=args.lang, iqr_multiplier=args.iqr
                )
            else:
                train_univariate.train_and_evaluate_by_char_count_scaled(
                    file_paths=file_paths, language=args.lang
                )
        
        elif args.model_type == 'multivariate':
            print("Training multivariate model...")
            if args.preprocessed:
                train_multivariate.train_evaluate_multi_var_preprocessed_scaled(
                    file_paths=file_paths, language=args.lang, iqr_multiplier=args.iqr
                )
            else:
                train_multivariate.train_evaluate_visualize_multi_var_scaled(
                    file_paths=file_paths, language=args.lang
                )
        print("Training complete. Check 'results' and 'saved_models' folders.")

if __name__ == "__main__":
    main()