# Simple Dense Retrieval

This project implements a **Dense Retrieval** system, focusing on efficient document retrieval tasks using the **Amazon ESCI dataset**. The model utilizes **dual encoders** to separately encode queries and documents, providing efficient and scalable retrieval in large-scale datasets.

## Features

- **Dual Encoder Architecture**: Separate encoders for queries and documents, enabling efficient and scalable dense retrieval.
- **Cosine Similarity-based Contrastive Learning**: The model uses contrastive learning with cosine similarity to learn effective query and document representations.
- **Customizable and Extensible**: Support for saving and loading custom layers and model architectures, making it easy to extend or adapt for different use cases.
- **Efficient Retrieval**: Designed for encoding and indexing large document corpora, providing fast and scalable retrieval capabilities.

## Installation

### Requirements

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- PyTorch
- Hugging Face Transformers
- Pandas
- NumPy

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/marevol/simple-dense-retrieval.git
   cd simple-dense-retrieval
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

   This will create a virtual environment and install all the necessary dependencies listed in `pyproject.toml`.

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Data Preparation

This project relies on the **Amazon ESCI dataset** for training and evaluating the dense retrieval model. You need to download the dataset and place it in the correct directory.

1. **Download the dataset**:
   - Obtain the `shopping_queries_dataset_products.parquet` and `shopping_queries_dataset_examples.parquet` files from the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data).

2. **Place the downloaded files** in the `downloads` directory within your project folder:
   ```bash
   ./downloads/shopping_queries_dataset_products.parquet
   ./downloads/shopping_queries_dataset_examples.parquet
   ```

3. **Verify data paths**:
   - The `main.py` script is set to load the dataset from the `downloads` directory by default. If you place the files elsewhere, modify the paths in the script accordingly.

## Usage

### Running the Training Script

The `main.py` script demonstrates how to use the **Amazon ESCI dataset** to train the dense retrieval model and evaluate its performance.

To run the training and evaluation:

```bash
poetry run python main.py
```

This script performs the following steps:

1. **Data Loading**: Loads product titles and queries from the Amazon ESCI dataset, pairing queries with their corresponding product titles using the `exact` label.
2. **Model Initialization**: Initializes dual encoders for queries and documents, using a pre-trained transformer model such as `intfloat/multilingual-e5-small`.
3. **Training**: The model is trained using contrastive learning with positive and negative document pairs. The query and document embeddings are optimized separately using the cosine similarity loss.
4. **Evaluation**: The trained model is evaluated on the test set, and performance metrics such as accuracy and loss are calculated.

### File Structure

- `main.py`: The main script for training and evaluating the dense retrieval model using the Amazon ESCI dataset.
- `simple_dense_retrieval/model.py`: Defines the `QueryEncoder` and `DocumentEncoder` classes for encoding queries and documents using pre-trained transformer models.
- `simple_dense_retrieval/train.py`: Implements the training process, including contrastive loss calculation and optimization steps.
- `simple_dense_retrieval/evaluate.py`: Contains functions to evaluate the model's performance on the test dataset.

### Model Saving and Loading

The project supports saving and loading models, tokenizers, and additional custom parameters such as `max_length`. The model is saved in both `state_dict` format and Hugging Face's `pretrained` format, allowing for flexibility and reusability. The script saves the following:

1. **Model Weights**: Saved using `state_dict` and Hugging Face's `save_pretrained` method.
2. **Tokenizers**: Saved using Hugging Face's `save_pretrained` method.
3. **Custom Parameters**: Parameters such as `max_length` are saved in a JSON file.

### Example of Saving the Model

```bash
poetry run python main.py
```

Upon completion of training:

1. **Model Saving**: The trained model is saved in the `dr_model` directory, including both `state_dict` and `pretrained` formats.
2. **Logging**: Training and evaluation logs, including loss and accuracy metrics, are saved in `train_dr.log`.
3. **Console Output**: Key performance metrics and progress are printed to the console.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.

