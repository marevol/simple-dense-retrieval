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
    def __init__(self, model, tokenizer, max_length=512):
        super(DocumentEncoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length

    def forward(self, document_texts):
        # Tokenize document texts
        encoded_inputs = self.tokenizer(
            document_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded_inputs["input_ids"].to(self.model.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.model.device)

        # Generate document embeddings
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        document_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
        return document_embeddings
