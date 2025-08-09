import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
import math

class Statistics:
    def __init__(self):
        self.n=0
        self.mean=0
        self.M2 = 0.0  # Sum of squares of differences from the current mean

    def update(self, x):
        '''Updates the mean and variance with a new data point x.'''
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self):
        '''Returns the variance based on the values seen so far.'''
        return self.M2 / self.n if self.n > 1 else 0.0
    
    @property
    def std(self):
        '''Returns the standard deviation (square root of variance).'''
        return math.sqrt(self.variance)

class DataLoader:
    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        self.bounds = {}

    def _calculate_iqr_bounds(self, iqr_multiplier: float, language: str = 'all'):
        '''Computes IQR-based lower and upper bounds for token/word length ratio.'''
        bounds_key = (language, iqr_multiplier)
        if bounds_key in self.bounds:
            return
        all_ratios = []
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            desc = file_path.split('\\')[-1].split('/')[-1]
            columns_to_load = ["content", "tiktoken_r50k_base_len"]
            if language != 'all':
                columns_to_load.append("language")

            for batch in tqdm(parquet_file.iter_batches(batch_size=8192, columns=columns_to_load), desc=f"Calculating ratios for '{language}' on {desc}"):
                chunk = batch.to_pandas()
                target_chunk = chunk
                if language != 'all':
                    target_chunk = chunk[chunk['language'] == language]
                word_counts = target_chunk['content'].astype(str).str.split().str.len()
                valid_mask = word_counts > 0
                if valid_mask.any():
                    ratios = target_chunk['tiktoken_r50k_base_len'][valid_mask] / word_counts[valid_mask]
                    all_ratios.extend(ratios.dropna().tolist())
        ratios_series = pd.Series(all_ratios)
        Q1 = ratios_series.quantile(0.25)
        Q3 = ratios_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (iqr_multiplier * IQR)
        upper_bound = Q3 + (iqr_multiplier * IQR)
        self.bounds[bounds_key] = (lower_bound, upper_bound)

    def load_preprocessed_data(self, iqr_multiplier: float = 1.5, language: str = 'all'):
        '''Yields filtered data chunks where token/word ratios fall within the calculated IQR bounds.'''
        bounds_key = (language, iqr_multiplier)
        if bounds_key not in self.bounds:
            self._calculate_iqr_bounds(iqr_multiplier, language)
        lower_bound, upper_bound = self.bounds[bounds_key]
        columns_to_load = ["content", "tiktoken_r50k_base_len"]
        if language != 'all':
            columns_to_load.append("language")

        for chunk in self.load_raw_data(columns=columns_to_load):
            target_chunk = chunk
            if language != 'all':
                target_chunk = chunk[chunk['language'] == language].copy()

            word_counts = target_chunk['content'].astype(str).str.split().str.len()
            ratios = target_chunk['tiktoken_r50k_base_len']/word_counts
            mask = ratios.between(lower_bound, upper_bound)
            yield target_chunk[mask]

    def load_raw_data(self, columns: list = None, chunk_size: int = 2048):
        '''Loads raw data from parquet files in chunks as pandas DataFrames.'''
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            desc = file_path.split('\\')[-1].split('/')[-1]
            for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size, columns=columns), desc=f"Loading raw from {desc}"):
                yield batch.to_pandas()

    def get_column_names(self):
        '''Returns the list of column names.'''
        return pq.ParquetFile(self.file_paths[0]).schema.names