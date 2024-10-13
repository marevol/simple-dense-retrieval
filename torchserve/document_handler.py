import logging
import pickle

import torch
from transformers import AutoModel, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

from simple_dense_retrieval.model import DocumentEncoder

logger = logging.getLogger(__name__)


class DocumentEncoderHandler(BaseHandler):
    def initialize(self, ctx):
        """
        Initialize the model and tokenizer. This method is called once during the start of the service.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the directory where the model artifacts are stored
        model_dir = ctx.system_properties.get("model_dir")

        # Load the tokenizer and the pretrained model
        logger.info(f"Loading tokenizer from {model_dir}/document_encoder")
        doc_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/document_encoder")
        logger.info(f"Loading model from {model_dir}/document_encoder")
        doc_model = AutoModel.from_pretrained(f"{model_dir}/document_encoder")

        # Load encoders for brand and color
        with open(f"{model_dir}/brand_encoder.pkl", "rb") as f:
            brand_encoder = pickle.load(f)
        with open(f"{model_dir}/color_encoder.pkl", "rb") as f:
            color_encoder = pickle.load(f)

        # Initialize the DocumentEncoder model
        self.document_encoder = DocumentEncoder(doc_model, doc_tokenizer, brand_encoder, color_encoder).to(self.device)

        self.document_encoder.load_state_dict(torch.load(f"{model_dir}/encoder.pth", map_location=self.device))
        logger.info("DocumentEncoder model loaded successfully")
        self.document_encoder.eval()  # Set the model to evaluation mode
        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess the incoming data. Extract titles, brands, and colors from the request.
        Expected input format: [{"title": "...", "brand": "...", "color": "..."}, ...]
        """
        titles = []
        brands = []
        colors = []

        for item in data[0].get("body", []):
            titles.append(item.get("title", ""))
            brands.append(item.get("brand", "Unknown"))  # Default to "Unknown" if brand is missing
            colors.append(item.get("color", "Unknown"))  # Default to "Unknown" if color is missing

        return titles, brands, colors

    def inference(self, input_data):
        """
        Perform inference on the batch of document titles, brands, and colors using the DocumentEncoder model.
        """
        titles, brands, colors = input_data

        # Perform batch inference
        with torch.no_grad():  # Disable gradient computation for inference
            document_embeddings = self.document_encoder(titles, brands, colors).detach().cpu().numpy()

        return document_embeddings

    def postprocess(self, inference_output):
        """
        Postprocess the inference result and return it in a format suitable for the client.
        """
        # Convert the inference output (embedding vectors) to a list of lists
        return [inference_output.tolist()]
