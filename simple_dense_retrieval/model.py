import torch


class QueryEncoder(torch.nn.Module):
    def __init__(self, model, tokenizer, max_length=512):
        super(QueryEncoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length

    def forward(self, query_texts):
        # Tokenize query texts
        encoded_inputs = self.tokenizer(
            query_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded_inputs["input_ids"].to(self.model.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.model.device)

        # Generate query embeddings
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        query_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
        return query_embeddings


class DocumentEncoder(torch.nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        brand_encoder,
        color_encoder,
        brand_embedding_dim=16,
        color_embedding_dim=16,
        output_dim=384,
        max_length=512,
    ):
        super(DocumentEncoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.brand_encoder = brand_encoder
        self.color_encoder = color_encoder
        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length

        # Embedding layers for brand and color
        self.brand_embedding = torch.nn.Embedding(len(brand_encoder), brand_embedding_dim, padding_idx=0)
        self.color_embedding = torch.nn.Embedding(len(color_encoder), color_embedding_dim, padding_idx=0)

        # Combine brand, color, and document embeddings
        self.fc = torch.nn.Linear(
            self.hidden_size + brand_embedding_dim + color_embedding_dim, output_dim
        )  # Output size can be adjusted

    def forward(self, document_texts, product_brands, product_colors):
        device = self.model.device

        # Tokenize document texts
        encoded_inputs = self.tokenizer(
            document_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        # Generate document embeddings
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        document_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

        # Label encode brands and colors before embedding
        brand_ids = self.brand_encoder.transform(product_brands)
        encoded_brands = torch.tensor(brand_ids).to(device)
        color_ids = self.color_encoder.transform(product_colors)
        encoded_colors = torch.tensor(color_ids).to(device)

        # Pass label encoded brands and colors through their respective embeddings
        brand_embeddings = self.brand_embedding(encoded_brands)
        color_embeddings = self.color_embedding(encoded_colors)

        # Concatenate document embeddings with brand and color embeddings
        combined_embeddings = torch.cat([document_embeddings, brand_embeddings, color_embeddings], dim=1)

        # Final projection
        final_embeddings = self.fc(combined_embeddings)
        return final_embeddings
