# Token Estimator

This project provides a command-line interface (CLI) tool to estimate the number of tokens for a given text prompt. It features single-variable and multi-variable linear regression models based on the `r50k_base` tokenizer, which is used by older OpenAI models.

## Core Features

- **Single-Variable Token Estimation**: Predicts token count based on character length.
- **Multi-Variable Token Estimation**: Predicts token count based on character length and word count.
- **Model Training**: Supports training on both raw and preprocessed data (with outlier removal).
- **Data Analysis Tools**: Includes scripts to analyze dataset characteristics, such as language distribution and punctuation frequency.
- **Powerful CLI**: A robust command-line interface to run all project components, including data preparation, training, and analysis.
- **Clean & Modular Codebase**: Designed with a clean, modular structure for easy maintenance and extension.

## Project Structure

The repository is structured to be clean and focused on the source code.

```
Token-Estimator/
├── src/
│   ├── analysis/
│   │   └── analysis_plots.py
│   ├── data_utils/
│   │   └── data_loader.py
│   ├── models/
│   │   ├── univariate.py
│   │   └── multivariate.py
│   ├── tokenization/
│   │   └── r50k_tokenizer.py
│   ├── training/
│   │   ├── train_univariate.py
│   │   └── train_multivariate.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
First, clone the project from GitHub:
```bash
git clone https://github.com/M-Mahdi-Razmjoo/Token-Estimator.git
cd Token-Estimator```

### 2. Create and Activate a Virtual Environment
Using a virtual environment is highly recommended to manage project dependencies in isolation.

- **Create the environment:**
  ```bash
  python -m venv venv
  ```
- **Activate on Windows (Git Bash):**
  ```bash
  source venv/Scripts/activate
  ```
- **Activate on macOS and Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required libraries using the `requirements.txt` file:```bash
pip install -r requirements.txt
```

## Data Preparation (Required)

This repository does **not** include the dataset. You must obtain and set it up locally according to the following instructions.

1.  Create a directory on your local machine to store the dataset. For example:
    ```
    C:\my_datasets\token_estimator_data
    ```

2.  Inside that directory, create a subdirectory named `shuffled_data`.

3.  Place all your Parquet dataset files (`shuffled_merged_1.parquet`, `shuffled_merged_2.parquet`, etc.) inside the `shuffled_data` subdirectory. The final structure should look like this:
    ```
    C:\my_datasets\token_estimator_data\
    └── shuffled_data\
        ├── shuffled_merged_1.parquet
        ├── shuffled_merged_2.parquet
        └── ...
    ```

4.  When running the application, you must provide the path to this `shuffled_data` directory using the `--data-dir` flag.

## Usage

All project functionalities are accessible via the `main.py` script.

### Model Training

- **Train a single-variable model on raw data:**
  ```bash
  python main.py --data-dir "C:\path\to\your\shuffled_data" train univariate
  ```

- **Train a multi-variable model on preprocessed English data:**
  ```bash
  python main.py --data-dir "C:\path\to\your\shuffled_data" train multivariate --preprocessed --lang English
  ```

## Outputs

The application will automatically create the following directories in the project root to store its outputs:
- **`results/`**: All plots, images, and model evaluation metrics will be saved here.
- **`saved_models/`**: The trained model files (`.joblib`) will be saved here.

## License

This project is licensed under the MIT License.
