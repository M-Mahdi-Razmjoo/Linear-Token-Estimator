import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.data_utils.data_loader import DataLoader, Statistics 
import string
from collections import Counter
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FuncFormatter
from src.tokenization.r50k_tokenizer import Tokenizer

class DataMining:
    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        self.statistics = Statistics()
        
    def plot_language_comparison(self, languages_to_plot):
        '''Generates a log-log scatter plot comparing word counts and token counts across different languages.'''
        is_comparison_mode = isinstance(languages_to_plot, list)
        data_by_language = {}
        if is_comparison_mode:
            for lang in languages_to_plot:
                data_by_language[lang] = []
        else:
            data_by_language['all'] = []

        data_loader = DataLoader(self.file_paths)
        columns_to_load = ["content", "tiktoken_r50k_base_len", "language"]

        for chunk in tqdm(data_loader.load_raw_data(columns=columns_to_load), desc="Scanning files"):
            target_languages = languages_to_plot if is_comparison_mode else chunk['language'].unique()
            
            for lang in target_languages:
                if is_comparison_mode:
                    lang_chunk = chunk[chunk['language'] == lang]
                else:
                    lang_chunk = chunk

                word_counts = lang_chunk['content'].astype(str).str.split().str.len()
                token_counts = lang_chunk['tiktoken_r50k_base_len']

                points = list(zip(word_counts, token_counts))

                if is_comparison_mode:
                    data_by_language[lang].extend(points)
                else:
                    data_by_language['all'].extend(points)

        plt.figure(figsize=(16, 9))

        color_cycle = plt.cm.get_cmap('Dark2', len(languages_to_plot) if is_comparison_mode else 1)

        for i, (lang, points) in enumerate(data_by_language.items()):
            plot_data = points
            x_samples, y_samples = zip(*plot_data)

            plot_color = 'darkblue' if not is_comparison_mode else color_cycle(i)
            plt.scatter(x_samples, y_samples, alpha=0.05, s=1, color=plot_color, label=lang, edgecolors='none')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Word Count', fontsize=14)
        plt.ylabel('Token Count', fontsize=14)
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        
        if is_comparison_mode:
            plt.legend(markerscale=5, fontsize=12) # Increased markerscale for better visibility in legend
            
        filename = "language_comparison_plot.png" if is_comparison_mode else "all_data_scatter_plot_full.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()

    def visualize_preprocessing_effect(self, std_dev_threshold=3.0, language: str = 'English'):   
        '''Visualizes the effect of outlier removal based on token-to-word ratio.'''     
        self.statistics = Statistics()
        all_points = []
        
        data_loader = DataLoader(self.file_paths)
        
        columns_to_load = ["content", "tiktoken_r50k_base_len"]
        if language != 'all':
            columns_to_load.append("language")
        
        raw_data_generator = data_loader.load_raw_data(columns=columns_to_load)

        for chunk in tqdm(raw_data_generator, desc="Scanning files"):
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()
            else:
                target_chunk = chunk

            word_counts = target_chunk['content'].astype(str).str.split().str.len()
            token_counts = target_chunk['tiktoken_r50k_base_len']
            
            for wc, tc in zip(word_counts, token_counts):
                all_points.append((wc, tc))

            valid_mask = word_counts > 0
            if valid_mask.any():
                ratios = token_counts[valid_mask] / word_counts[valid_mask]
                for ratio in ratios:
                    self.statistics.update(ratio)

        mean_ratio = self.statistics.mean
        std_ratio = self.statistics.std
        
        lower_bound = mean_ratio - (std_dev_threshold * std_ratio)
        upper_bound = mean_ratio + (std_dev_threshold * std_ratio)

        all_x, all_y = zip(*all_points)
        
        inlier_x, inlier_y = [], []
        for wc, tc in all_points:
            if wc > 0 and lower_bound <= (tc / wc) <= upper_bound:
                inlier_x.append(wc)
                inlier_y.append(tc)

        scope_filename = language.lower() if language != 'all' else 'all_languages'

        if all_x:
            plt.figure(figsize=(16, 9))
            plt.xscale('log')
            plt.yscale('log')
            plt.scatter(all_x, all_y, alpha=0.05, s=1, color='darkblue')
            plt.xlabel('Word Count', fontsize=14)
            plt.ylabel('Token Count', fontsize=14)
            plt.grid(True, which="both", linestyle='--', linewidth=0.5)
            plot_filename_before = f"visualization_{scope_filename}_before.png"
            plt.savefig(plot_filename_before, dpi=150, bbox_inches='tight')
            plt.close()

        if inlier_x:
            plt.figure(figsize=(16, 9))
            plt.xscale('log')
            plt.yscale('log')
            plt.scatter(inlier_x, inlier_y, alpha=0.1, s=2, color='darkgreen')
            plt.xlabel('Word Count', fontsize=14)
            plt.ylabel('Token Count', fontsize=14)
            plt.grid(True, which="both", linestyle='--', linewidth=0.5)
            plot_filename_after = f"visualization_{scope_filename}_after.png"
            plt.savefig(plot_filename_after, dpi=150, bbox_inches='tight')
            plt.close()

    def plot_punctuation_histogram(self, language: str = 'English', threshold_percent=0.05):
        '''Creates a histogram showing how frequently different punctuation counts appear in prompts. Filters out low-frequency values based on a percentage threshold.'''
        punctuation_counts = []
        
        data_loader = DataLoader(self.file_paths)
        punctuation_set = set(string.punctuation)
        columns_to_load = ["content"]
        if language != 'all':
            columns_to_load.append("language")
        raw_data_generator = data_loader.load_raw_data(columns=columns_to_load)

        for chunk in tqdm(raw_data_generator, desc="Counting punctuation"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()
            for text in target_chunk['content']:
                if isinstance(text, str):
                    punc_count = sum(1 for char in text if char in punctuation_set)
                    punctuation_counts.append(punc_count)

        total_prompts = len(punctuation_counts)
        count_distribution = Counter(punctuation_counts)
        filtered_counts = {}
        for punc_count, freq in count_distribution.items():
            percentage = (freq / total_prompts) * 100
            if percentage >= threshold_percent:
                filtered_counts[punc_count] = percentage

        sorted_items = sorted(filtered_counts.items())
        x_values, y_percentages = zip(*sorted_items)

        plt.figure(figsize=(16, 9))
        plt.bar(x_values, y_percentages, color='blue', alpha=0.8, edgecolor='black')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        plt.xlabel('Number of Punctuations', fontsize=14)
        plt.ylabel('Percentage', fontsize=14)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"punctuation_histogram_filtered_{filename_scope}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_punctuation_distribution_per_word(self, language: str = 'English', threshold_percent=0.05):
        '''Plots the distribution of punctuation-per-word ratios across prompts, highlighting common punctuation density levels.'''
        punctuation_counts = []

        bin_width = 0.05
        def bin_value(x):
            return round(bin_width * round(x / bin_width), 2)

        data_loader = DataLoader(self.file_paths)
        punctuation_set = set(string.punctuation)
        columns_to_load = ["content"]
        if language != 'all':
            columns_to_load.append("language")
        raw_data_generator = data_loader.load_raw_data(columns=columns_to_load)

        for chunk in tqdm(raw_data_generator, desc="Counting punctuation"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()
            for text in target_chunk['content']:
                if isinstance(text, str):
                    words = text.split()
                    if words:
                        punc_count = sum(1 for char in text if char in punctuation_set)
                        ratio = punc_count / len(words)
                        binned_ratio = bin_value(ratio)
                        punctuation_counts.append(binned_ratio)
                    else:
                        punctuation_counts.append(0)

        total_prompts = len(punctuation_counts)
        count_distribution = Counter(punctuation_counts)

        filtered_counts = {}
        for punc_count, freq in count_distribution.items():
            percentage = (freq / total_prompts) * 100
            if percentage >= threshold_percent:
                filtered_counts[punc_count] = percentage

        sorted_items = sorted(filtered_counts.items())
        x_values, y_percentages = zip(*sorted_items)

        plt.figure(figsize=(16, 9))
        plt.bar(x_values, y_percentages, width=bin_width * 0.9, color='blue', alpha=0.8, edgecolor='black')  
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        plt.xlabel('Punctuations Per Word', fontsize=14)
        plt.ylabel('Percentage', fontsize=14)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"punctuation_per_word_histogram_filtered_{filename_scope}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_word_tokenization_histogram(self, language: str = 'English'):
        '''Analyzes how many tokens each individual word generates using the tokenizer, and fits a power-law curve to the distribution.'''
        token_distribution = Counter()
        tokenizer = Tokenizer()
        data_loader = DataLoader(self.file_paths)
        columns_to_load = ["content"]
        if language != 'all':
            columns_to_load.append("language")
        raw_data_generator = data_loader.load_raw_data(columns=columns_to_load)
        for chunk in tqdm(raw_data_generator, desc="Analyzing Words"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()

            for text in target_chunk['content']:
                if isinstance(text, str):
                    words = text.split()
                    for word in words:
                        num_tokens = tokenizer.token_count(word)
                        token_distribution[num_tokens] += 1

        sorted_items = sorted(token_distribution.items())
        x_values = np.array([item[0] for item in sorted_items if item[0] > 0])
        y_frequencies = np.array([item[1] for item in sorted_items if item[0] > 0])
        
        log_x = np.log10(x_values)
        log_y = np.log10(y_frequencies)
        coeffs = np.polyfit(log_x, log_y, 1)
        power_law_exponent = -coeffs[0]
        power_law_constant = 10**coeffs[1]

        print(f"Exponent:{power_law_exponent:.4f}")
        print(f"Constant:{power_law_constant:.2f}")

        plt.figure(figsize=(16, 9))
        plt.scatter(x_values, y_frequencies, color='blue', alpha=0.7, label='Actual Frequency Data', s=50)
        x_fit = np.linspace(min(x_values), max(x_values), 200)
        y_fit = power_law_constant * (x_fit ** (-power_law_exponent))
        plt.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=3, label=f'Power-Law Fit')       
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: format(int(y), ',')))
        plt.xlabel('Number of Tokens Generated from a Single Word', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.xscale('log')
        plt.yscale('log')
        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"word_tokenization_power_law_{filename_scope}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_token_per_char_distribution(self, language: str = 'English'):
        '''Plots the relationship between character count and token count using log-log scale to explore how tokenization scales with character length.'''
        all_points = []
        data_loader = DataLoader(self.file_paths)
        columns_to_load = ["content", "tiktoken_r50k_base_len"]
        if language != 'all':
            columns_to_load.append("language")
            
        raw_data_generator = data_loader.load_raw_data(columns=columns_to_load)
        for chunk in tqdm(raw_data_generator, desc="Collecting data"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()
            char_counts = target_chunk['content'].astype(str).str.len()
            token_counts = target_chunk['tiktoken_r50k_base_len']
            valid_points = list(zip(char_counts, token_counts)) # Filter out points with zero counts for log scale compatibility
            all_points.extend([p for p in valid_points if p[0] > 0 and p[1] > 0])
            
        x_char_counts, y_token_counts = zip(*all_points)
        plt.figure(figsize=(16, 9))

        plt.scatter(x_char_counts, y_token_counts, alpha=0.05, s=5, edgecolors='none')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Character Count', fontsize=14)
        plt.ylabel('Token Count', fontsize=14)
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)

        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"token_char_scatter_log_{filename_scope}.png"
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
        plt.close()

    def plot_token_per_word_char_ratio(self, language: str = 'English'):
        '''Plots token count against the ratio of word count to character count in each prompt.'''
        all_points = []
        data_loader = DataLoader(self.file_paths)
        columns_to_load = ["content", "tiktoken_r50k_base_len"]
        if language != 'all':
            columns_to_load.append("language")
        raw_data_generator = data_loader.load_raw_data(columns=columns_to_load)
        for chunk in tqdm(raw_data_generator, desc="Collecting data"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()
            for index, row in target_chunk.iterrows():
                text = str(row['content'])
                char_count = len(text)
                word_count = len(text.split())
                token_count = row['tiktoken_r50k_base_len']
                if char_count > 0:
                    ratio = word_count / char_count
                    all_points.append((ratio, token_count))
        x_ratios, y_token_counts = zip(*all_points)
        plt.figure(figsize=(16, 9))
        plt.scatter(x_ratios, y_token_counts, alpha=0.05, s=5, edgecolors='none')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Ratio (Word Count / Character Count)', fontsize=14)
        plt.ylabel('Token Count', fontsize=14)
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"token_vs_word_char_ratio_{filename_scope}.png"
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
        plt.close()

    def plot_token_per_char_word_ratio(self, language: str = 'English'):
        '''Plots token count against the ratio of character count to word count in each prompt.'''
        all_points = []
        data_loader = DataLoader(self.file_paths)
        columns_to_load = ["content", "tiktoken_r50k_base_len"]
        if language != 'all':
            columns_to_load.append("language")
        raw_data_generator = data_loader.load_raw_data(columns=columns_to_load)
        for chunk in tqdm(raw_data_generator, desc="Collecting data"):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()
            for index, row in target_chunk.iterrows():
                text = str(row['content'])
                char_count = len(text)
                word_count = len(text.split())
                token_count = row['tiktoken_r50k_base_len']
                if word_count > 0:
                    ratio = char_count / word_count
                    all_points.append((ratio, token_count))
        x_ratios, y_token_counts = zip(*all_points)
        plt.figure(figsize=(16, 9))
        plt.scatter(x_ratios, y_token_counts, alpha=0.05, s=5, edgecolors='none')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Ratio (Character Count / Word Count)', fontsize=14)
        plt.ylabel('Token Count', fontsize=14)
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        filename_scope = language.lower().replace(" ", "_") if language != 'all' else 'all_data'
        plot_filename = f"token_vs_char_word_ratio_{filename_scope}.png"
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
        plt.close()