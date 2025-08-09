import os
import shutil
import gc
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from src.data_utils.data_loader import DataLoader

def shuffle_and_track_data(base_data_path: str, random_seed: int = 42):
    # Paths are now relative to the provided base_data_path
    INPUT_FILES = [os.path.join(base_data_path, f"merged_{i}.parquet") for i in range(1, 11)]
    TEMP_DIR = os.path.join(base_data_path, "temp_shuffle")
    OUTPUT_DIR_DATA = os.path.join(base_data_path, "shuffled_data")
    OUTPUT_DIR_MAP = os.path.join(base_data_path, "shuffled_map")
    
    TARGET_ROW_COUNTS = [795143] * 9 + [795151]
    NUM_TEMP_PARTITIONS = 100

    for path in [TEMP_DIR, OUTPUT_DIR_DATA, OUTPUT_DIR_MAP]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Check if input files exist
    for f in INPUT_FILES:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file not found: {f}. Please ensure original data is in '{base_data_path}'.")

    base_schema = pq.read_schema(INPUT_FILES[0])
    final_schema_list = list(base_schema)
    final_schema_list.append(pa.field('original_source_file', pa.string()))
    final_schema_list.append(pa.field('original_row_index', pa.int64()))
    final_schema = pa.schema(final_schema_list)

    writers = [pq.ParquetWriter(os.path.join(TEMP_DIR, f'part-{i}.parquet'), schema=final_schema) for i in range(NUM_TEMP_PARTITIONS)]
    rng = np.random.RandomState(random_seed)
    total_rows_processed = 0
    pbar_phase1 = tqdm(total=sum(TARGET_ROW_COUNTS), desc="Partitioning", unit="row")
    try:
        for file_path in INPUT_FILES:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=8192):
                chunk_df = batch.to_pandas()
                chunk_df['original_source_file'] = os.path.basename(file_path)
                chunk_df['original_row_index'] = np.arange(total_rows_processed, total_rows_processed + len(chunk_df))
                partition_indices = rng.randint(0, NUM_TEMP_PARTITIONS, size=len(chunk_df))
                chunk_df['temp_partition'] = partition_indices
                for i, group in chunk_df.groupby('temp_partition'):
                    group_to_write = group.drop(columns=['temp_partition'])
                    table = pa.Table.from_pandas(group_to_write, schema=final_schema, preserve_index=False)
                    writers[i].write_table(table)
                pbar_phase1.update(len(chunk_df))
                total_rows_processed += len(chunk_df)
    finally:
        for writer in writers:
            writer.close()
        pbar_phase1.close()

    temp_files = [os.path.join(TEMP_DIR, f'part-{i}.parquet') for i in range(NUM_TEMP_PARTITIONS)]
    random.Random(random_seed).shuffle(temp_files) 

    current_file_index = 0
    rows_written_to_current_file = 0
    data_buffer, map_buffer = [], []
    
    pbar_phase2 = tqdm(total=sum(TARGET_ROW_COUNTS), desc="Writing", unit="row")
    
    for temp_file_path in temp_files:
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0: continue
        parquet_file = pq.ParquetFile(temp_file_path)
        for batch in parquet_file.iter_batches(batch_size=8192):
            chunk_df = batch.to_pandas()
            original_columns = [c for c in chunk_df.columns if c not in ['original_source_file', 'original_row_index']]
            data_chunk = chunk_df[original_columns]
            map_chunk = chunk_df[['original_source_file', 'original_row_index']]
            rows_to_process = len(data_chunk)
            processed_idx = 0
            while processed_idx < rows_to_process:
                if current_file_index >= len(TARGET_ROW_COUNTS): break
                target_rows = TARGET_ROW_COUNTS[current_file_index]
                rows_needed = target_rows - rows_written_to_current_file
                rows_to_take = min(rows_needed, rows_to_process - processed_idx)

                data_slice = data_chunk.iloc[processed_idx : processed_idx + rows_to_take]
                map_slice = map_chunk.iloc[processed_idx : processed_idx + rows_to_take]
                data_buffer.append(data_slice); map_buffer.append(map_slice)
                
                rows_written_to_current_file += len(data_slice)
                processed_idx += len(data_slice)
                pbar_phase2.update(len(data_slice))

                if rows_written_to_current_file >= target_rows:
                    full_data_df = pd.concat(data_buffer, ignore_index=True)
                    full_map_df = pd.concat(map_buffer, ignore_index=True)

                    shuffled_indices = np.arange(len(full_data_df))
                    np.random.RandomState(random_seed + current_file_index).shuffle(shuffled_indices) 

                    shuffled_data_to_write = full_data_df.iloc[shuffled_indices].reset_index(drop=True)
                    shuffled_map_to_write = full_map_df.iloc[shuffled_indices].reset_index(drop=True)

                    shuffled_data_to_write.to_parquet(os.path.join(OUTPUT_DIR_DATA, f"shuffled_merged_{current_file_index + 1}.parquet"), index=False)
                    shuffled_map_to_write.to_parquet(os.path.join(OUTPUT_DIR_MAP, f"map_for_shuffled_merged_{current_file_index + 1}.parquet"), index=False)
                    
                    current_file_index += 1
                    rows_written_to_current_file = 0
                    data_buffer, map_buffer = [], []
            if current_file_index >= len(TARGET_ROW_COUNTS): break
        
        del parquet_file
        gc.collect()

    if data_buffer:
        full_data_df = pd.concat(data_buffer, ignore_index=True)
        full_map_df = pd.concat(map_buffer, ignore_index=True)
        shuffled_indices = np.arange(len(full_data_df))
        np.random.RandomState(random_seed + current_file_index).shuffle(shuffled_indices)
        shuffled_data_to_write = full_data_df.iloc[shuffled_indices].reset_index(drop=True)
        shuffled_map_to_write = full_map_df.iloc[shuffled_indices].reset_index(drop=True)
        shuffled_data_to_write.to_parquet(os.path.join(OUTPUT_DIR_DATA, f"shuffled_merged_{current_file_index + 1}.parquet"), index=False)
        shuffled_map_to_write.to_parquet(os.path.join(OUTPUT_DIR_MAP, f"map_for_shuffled_merged_{current_file_index + 1}.parquet"), index=False)
        
    pbar_phase2.close()
    gc.collect()
    shutil.rmtree(TEMP_DIR)
    
def analyze_symbol_percentage_distribution(file_paths: list):
    SYMBOLS_TO_COUNT = {'=', '<', '>', '+', '-', '*', '/', '{', '}', '(', ')', ';'}
    prompt_symbol_counts = []
    data_loader = DataLoader(file_paths)
    data_generator = data_loader.load_raw_data(columns=["content"])
    for chunk in tqdm(data_generator, desc="Counting symbols"):
        chunk.dropna(subset=['content'], inplace=True)
        for text in chunk['content']:
            text = str(text)
            count = sum(1 for char in text if char in SYMBOLS_TO_COUNT)
            prompt_symbol_counts.append(count)

    total_prompts = len(prompt_symbol_counts)
    counts_frequency = Counter(prompt_symbol_counts)

    significant_counts = []
    for symbol_count, frequency in counts_frequency.items():
        percentage = (frequency / total_prompts) * 100
        if percentage >= 0.5:
            significant_counts.append((symbol_count, percentage))

    significant_counts.sort()
    x_values, y_percentages = zip(*significant_counts)
    plt.figure(figsize=(16, 9))
    bars = plt.bar(x_values, y_percentages, color='blue', edgecolor='black', zorder=2)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.xlabel('Number of symbols in a prompt', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.xticks(x_values)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, zorder=1)
    plot_filename = "results/symbol_count_percentage_distribution.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()