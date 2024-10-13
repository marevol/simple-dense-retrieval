import logging

import torch
from transformers import AutoModel, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

from simple_dense_retrieval.model import QueryEncoder

logger = logging.getLogger(__name__)


class QueryEncoderHandler(BaseHandler):
    def initialize(self, ctx):
        """
        Initialize the model and tokenizer. This method is called once during the start of the service.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the directory where the model artifacts are stored
        model_dir = ctx.system_properties.get("model_dir")

        # Load the tokenizer and the pretrained model
        logger.info(f"Loading tokenizer from {model_dir}/query_encoder")
        query_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/query_encoder")
        logger.info(f"Loading model from {model_dir}/query_encoder")
        query_model = AutoModel.from_pretrained(f"{model_dir}/query_encoder")

        # Initialize the QueryEncoder model
        self.query_encoder = QueryEncoder(query_model, query_tokenizer).to(self.device)
        self.query_encoder.load_state_dict(torch.load(f"{model_dir}/encoder.pth", map_location=self.device))
        logger.info("QueryEncoder model loaded successfully")
        self.query_encoder.eval()  # Set the model to evaluation mode
        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess the incoming data. In this case, extract the query text from the request.
        """
        return data[0].get("body", {"query": ""}).get("query", "")

    def inference(self, input_data):
        """
        Perform inference on the input query using the QueryEncoder model.
        """
        with torch.no_grad():  # Disable gradient calculation for inference
            query_embedding = self.query_encoder([input_data]).detach().cpu().numpy()
        return query_embedding

    def postprocess(self, inference_output):
        """
        Postprocess the inference result before returning it to the client.
        """
        return inference_output.tolist()
