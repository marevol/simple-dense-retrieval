import torch
import torch.nn.functional as F


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

        # Linear layers for projecting attributes and title
        self.title_proj = torch.nn.Linear(self.hidden_size, output_dim)
        self.attr_proj = torch.nn.Linear(brand_embedding_dim + color_embedding_dim, output_dim)

        # Final output layer
        self.fc = torch.nn.Linear(output_dim * 2, output_dim)

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

        # Generate document (title) embeddings
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        document_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

        # Encode brands and colors
        brand_ids = self.brand_encoder.transform(product_brands)
        encoded_brands = torch.tensor(brand_ids).to(device)
        color_ids = self.color_encoder.transform(product_colors)
        encoded_colors = torch.tensor(color_ids).to(device)

        # Pass through embedding layers for brand and color
        brand_embeddings = self.brand_embedding(encoded_brands)
        color_embeddings = self.color_embedding(encoded_colors)
        attr_embeddings = torch.cat([brand_embeddings, color_embeddings], dim=1)

        # Project attributes and title
        attr_proj = self.attr_proj(attr_embeddings)
        title_proj = self.title_proj(document_embeddings)

        # Concatenate title and attribute projections
        combined_embeddings = torch.cat([title_proj, attr_proj], dim=1)

        # Final projection
        final_embeddings = self.fc(combined_embeddings)
        return final_embeddings
