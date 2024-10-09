import logging
import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from simple_dense_retrieval.encoder import LabelEncoder
from simple_dense_retrieval.evaluate import evaluate_dense_retrieval
from simple_dense_retrieval.model import DocumentEncoder, QueryEncoder
from simple_dense_retrieval.train import train_dense_retrieval


def setup_logger():
    """
    Set up the logger for logging training and evaluation information.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("train_dr.log")],
    )


def drop_insufficient_data(df):
    """
    Drop queries that do not have enough positive examples.

    Args:
        df (DataFrame): DataFrame containing query and document information.

    Returns:
        DataFrame: Filtered DataFrame with sufficient positive examples.
    """
    id_df = df[["query_id", "exact"]]
    id_df.loc[:, ["total"]] = 1
    id_df = id_df.groupby("query_id").sum().reset_index()
    id_df = id_df[(id_df["exact"] > 0) & (id_df["total"] > id_df["exact"])]
    return pd.merge(id_df[["query_id"]], df, how="left", on="query_id")


def create_encoder(df, column_name, min_frequency=10):
    """
    Create an LabelEncoder for a specific column in the DataFrame.
    Categories with a frequency less than min_frequency will be considered infrequent and grouped together.
    Unknown values during transformation will be encoded as -1.
    Rows with NaN values in the specified column will be dropped.

    Args:
        df (DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to encode.
        min_frequency (int): The minimum frequency for a category to be treated as known.

    Returns:
        LabelEncoder: The fitted LabelEncoder.
    """
    # Drop rows with NaN in the specified column
    df_dropped = df[[column_name]].dropna()

    # Create LabelEncoder with min_frequency and unknown values set to 0
    encoder = LabelEncoder(min_frequency=min_frequency)

    # Fit the encoder on the column without NaN values
    encoder.fit(df_dropped[column_name].values)

    return encoder


def load_data():
    """
    Load and preprocess the dataset.

    Returns:
        DataFrame: Training DataFrame.
        DataFrame: Test DataFrame.
    """
    # Load datasets (modify this to your dataset path)
    product_df = pd.read_parquet("downloads/shopping_queries_dataset_products.parquet")
    example_df = pd.read_parquet("downloads/shopping_queries_dataset_examples.parquet")

    # Merge datasets and create a column for positive/negative label (1 for positive, 0 for negative)
    df = pd.merge(
        example_df[["example_id", "query_id", "product_id", "query", "esci_label", "split"]],
        product_df[["product_id", "product_title", "product_brand", "product_color"]],
        how="left",
        on="product_id",
    )[["example_id", "query_id", "query", "product_title", "product_brand", "product_color", "esci_label", "split"]]

    # Label exact matches as positive (1), everything else as negative (0)
    df["exact"] = df.esci_label.apply(lambda x: 1 if x == "E" else 0)

    # Drop queries without sufficient positive examples
    train_df = drop_insufficient_data(df[df.split == "train"])
    test_df = drop_insufficient_data(df[df.split == "test"])

    return train_df, test_df


class QueryDocumentDataset(Dataset):
    def __init__(self, df):
        """
        Dataset for query, positive, and negative document pairs.

        Args:
            df (DataFrame): DataFrame containing query, positive, and negative document information.
        """
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query = self.df.loc[idx, "query"]
        query_id = self.df.loc[idx, "query_id"]
        exact = self.df.loc[idx, "exact"]
        if exact == 1:
            positive_title = self.df.loc[idx, "product_title"]  # Positive document
            positive_brand = self.df.loc[idx, "product_brand"]
            positive_color = self.df.loc[idx, "product_color"]
            neg_df = self.df[(self.df["query_id"] == query_id) & (self.df["exact"] == 0)]
            negative_doc = neg_df.sample(1)  # Random negative document
            negative_title = negative_doc.iloc[0]["product_title"]
            negative_brand = negative_doc.iloc[0]["product_brand"]
            negative_color = negative_doc.iloc[0]["product_color"]
        else:
            pos_df = self.df[(self.df["query_id"] == query_id) & (self.df["exact"] == 1)]
            positive_doc = pos_df.sample(1)  # Random positive document
            positive_title = positive_doc.iloc[0]["product_title"]
            positive_brand = positive_doc.iloc[0]["product_brand"]
            positive_color = positive_doc.iloc[0]["product_color"]
            negative_title = self.df.loc[idx, "product_title"]  # Negative document
            negative_brand = self.df.loc[idx, "product_brand"]
            negative_color = self.df.loc[idx, "product_color"]

        # Each input text should start with "query: " or "passage: ", even for non-English texts.
        # For tasks other than retrieval, you can simply use the "query: " prefix.
        # https://huggingface.co/intfloat/multilingual-e5-small
        return {
            "query": f"query: {query}",
            "positive_title": f"passage: {positive_title}",
            "positive_brand": positive_brand if positive_brand is not None else "",
            "positive_color": positive_color if positive_color is not None else "",
            "negative_title": f"passage: {negative_title}",
            "negative_brand": negative_brand if negative_brand is not None else "",
            "negative_color": negative_color if negative_color is not None else "",
        }


def get_query_encoder(model_name, device="cpu"):
    """
    Get the query encoder model.

    Args:
        model_name (str): Pretrained model name.

    Returns:
        QueryEncoder: Query encoder model.
    """
    if model_name is not None and os.path.exists(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        encoder = QueryEncoder(model=model, tokenizer=tokenizer)
        encoder.load_state_dict(torch.load(os.path.join(model_name, "encoder.pth"), map_location=device))
        return encoder.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return QueryEncoder(model=model, tokenizer=tokenizer).to(device)


def get_document_encoder(model_name, brand_encoder=None, color_encoder=None, device="cpu"):
    """
    Get the document encoder model.

    Args:
        model_name (str): Pretrained model name.
        brand_encoder (LabelEncoder): Optional pre-trained brand encoder.
        color_encoder (LabelEncoder): Optional pre-trained color encoder.
        device (str): The device to load the model on.

    Returns:
        DocumentEncoder: Document encoder model.
    """
    if model_name is not None and os.path.exists(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        brand_encoder_path = os.path.join(model_name, "brand_encoder.pkl")
        with open(brand_encoder_path, "rb") as f:
            brand_encoder = pickle.load(f)
        color_encoder_path = os.path.join(model_name, "color_encoder.pkl")
        with open(color_encoder_path, "rb") as f:
            color_encoder = pickle.load(f)
        encoder = DocumentEncoder(
            model=model, tokenizer=tokenizer, brand_encoder=brand_encoder, color_encoder=color_encoder
        )
        encoder.load_state_dict(torch.load(os.path.join(model_name, "encoder.pth"), map_location=device))
        return encoder.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return DocumentEncoder(
        model=model, tokenizer=tokenizer, brand_encoder=brand_encoder, color_encoder=color_encoder
    ).to(device)


def save_models(query_encoder, document_encoder, optimizer, save_directory):
    """
    Save the trained models and optimizer.

    Args:
        query_encoder (QueryEncoder): Trained query encoder model.
        document_encoder (DocumentEncoder): Trained document encoder model.
        optimizer (Optimizer): Optimizer used during training.
        tokenizer (AutoTokenizer): Tokenizer used for encoding.
        save_directory (str): Directory to save the model.
    """
    query_encoder_path = os.path.join(save_directory, "query_encoder")
    if not os.path.exists(query_encoder_path):
        os.makedirs(query_encoder_path)
    document_encoder_path = os.path.join(save_directory, "document_encoder")
    if not os.path.exists(document_encoder_path):
        os.makedirs(document_encoder_path)

    query_encoder.model.save_pretrained(query_encoder_path)
    query_encoder.tokenizer.save_pretrained(query_encoder_path)
    torch.save(query_encoder.state_dict(), os.path.join(query_encoder_path, "encoder.pth"))

    document_encoder.model.save_pretrained(document_encoder_path)
    document_encoder.tokenizer.save_pretrained(document_encoder_path)
    brand_encoder_path = os.path.join(document_encoder_path, "brand_encoder.pkl")
    with open(brand_encoder_path, "wb") as f:
        pickle.dump(document_encoder.brand_encoder, f)
    color_encoder_path = os.path.join(document_encoder_path, "color_encoder.pkl")
    with open(color_encoder_path, "wb") as f:
        pickle.dump(document_encoder.color_encoder, f)
    torch.save(document_encoder.state_dict(), os.path.join(document_encoder_path, "encoder.pth"))

    optimizer_path = os.path.join(save_directory, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_path)


def train(logger, train_df, model_name="intfloat/multilingual-e5-small", save_directory="dr_model", device="cpu"):
    """
    Train the Dense Retrieval model.

    Args:
        logger (Logger): Logger for logging information.
        train_df (DataFrame): Training DataFrame.
        model_name (str): Pretrained model name.
        device (str): Device to train on.

    Returns:
        None
    """
    logger.info("Initializing encoders...")
    # Create encoders for brand and color
    brand_encoder = create_encoder(train_df, "product_brand", min_frequency=100)
    logger.info(f"Brand features: {len(brand_encoder)}")
    color_encoder = create_encoder(train_df, "product_color", min_frequency=100)
    logger.info(f"Color features: {len(color_encoder)}")

    logger.info("Initializing tokenizer and models...")
    query_encoder = get_query_encoder(model_name=model_name, device=device)
    document_encoder = get_document_encoder(
        model_name=model_name, brand_encoder=brand_encoder, color_encoder=color_encoder, device=device
    )

    logger.info("Preparing dataset and dataloader...")
    train_dataset = QueryDocumentDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(list(query_encoder.parameters()) + list(document_encoder.parameters()), lr=2e-5)

    logger.info("Starting training...")
    train_dense_retrieval(
        query_encoder,
        document_encoder,
        train_loader,
        optimizer,
        num_epochs=5,
        device=device,
    )

    logger.info("Saving trained models...")
    save_models(query_encoder, document_encoder, optimizer, save_directory)


def evaluate(logger, test_df, model_name="intfloat/multilingual-e5-small", save_directory="dr_model", device="cpu"):
    """
    Evaluate the trained Dense Retrieval model.

    Args:
        logger (Logger): Logger for logging information.
        test_df (DataFrame): Test DataFrame.
        model_name (str): Pretrained model name.
        device (str): Device to evaluate on.

    Returns:
        None
    """
    logger.info("Evaluating the model...")

    query_encoder_path = os.path.join(save_directory, "query_encoder")
    query_encoder = get_query_encoder(model_name=query_encoder_path, device=device)

    document_encoder_path = os.path.join(save_directory, "document_encoder")
    document_encoder = get_document_encoder(model_name=document_encoder_path, device=device)

    test_dataset = QueryDocumentDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    logger.info("Evaluating the model...")
    avg_loss, accuracy = evaluate_dense_retrieval(
        query_encoder,
        document_encoder,
        test_loader,
        device=device,
    )
    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "intfloat/multilingual-e5-small"

    logger.info("Loading data from dataset...")
    train_df, test_df = load_data()
    logger.info(f"Train data: {len(train_df)}, Test data: {len(test_df)}")

    train(logger, train_df, model_name=model_name, device=device)

    evaluate(logger, test_df, model_name=model_name, device=device)
