# Data Analysis

This directory contains visualizations and statistical summaries derived from the prompt dataset.

The dataset used in these analyses contained the following columns:

| conversation_hash | role | content | model | timestamp | language | state | country | hashed_ip | source | tiktoken_r50k_base_len | tiktoken_cl100k_base_len | tiktoken_o200k_base_len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

A PDF with the number of prompts per language is provided in language_prompt_counts.pdf

---

Below is a description of the files in this folder and what each visualization/report shows.

## 1. Punctuation distribution
- **`punctuation_distribution_all.png`**  
  Distribution of the number of punctuation marks per prompt for the entire dataset (e.g., percentage of prompts with 0, 1, 2, ... punctuation marks).
- **`punctuation_distribution_english.png`**  
  Same analysis restricted to English-language prompts.

## 2. Token distribution for single words
- **`token_distribution_single_word.png`**  
  Distribution of token counts for single words (words split by spaces in the prompt).  
  This figure is plotted on a semi-logarithmic scale and includes a fitted power-law curve.

## 3. Tokens vs. words (log–log)
- **`tokens_vs_words_all.png`**  
  Log–log plot of token count (vertical axis) versus word count (space-separated, horizontal axis) for the entire dataset. Useful to check scaling relations and heavy-tail behavior.
- **`tokens_vs_words_english.png`**  
  The same log–log relationship but restricted to English prompts.

## 4. Tokens vs. words — language comparison (log–log)
- **`tokens_vs_words_en_vs_zh.png`**  
  Log–log comparison of token count vs. word count for English (en) and Chinese (zh) prompts on the same axes to highlight cross-lingual differences.

## 5. Tokens vs. characters (log–log)
- **`tokens_vs_characters_all.png`**  
  Log–log plot showing token count versus character count for the entire dataset.
- **`tokens_vs_characters_english.png`**  
  Same plot restricted to English prompts.

---

### Notes
- Word counts are computed by splitting the `content` field on spaces.
- Character counts include all characters in the `content` field.  
- All plots were generated from the full dataset (no downsampling).  
