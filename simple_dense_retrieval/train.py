import logging

import torch.nn.functional as F


def contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=0.2):
    """
    Compute contrastive loss using cosine similarity for positive and negative pairs.

    Args:
        anchor_embeddings (Tensor): Anchor embeddings (e.g., query embeddings)
        positive_embeddings (Tensor): Positive pair embeddings
        negative_embeddings (Tensor): Negative pair embeddings
        margin (float): Margin for the contrastive loss (default: 0.2)

    Returns:
        Tensor: Loss value
    """
    # Compute cosine similarity
    pos_sim = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
    neg_sim = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)

    # Contrastive loss: maximize pos_sim and minimize neg_sim with a margin
    loss = F.relu(margin + neg_sim - pos_sim).mean()

    return loss, pos_sim, neg_sim


def train_dense_retrieval(
    query_encoder,
    document_encoder,
    train_loader,
    optimizer,
    num_epochs=3,
    device="cpu",
):
    """
    Train Dense Retrieval model with contrastive learning.

    Args:
        query_encoder (QueryEncoder): Query encoder model.
        document_encoder (DocumentEncoder): Document encoder model.
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer for training
        num_epochs (int): Number of epochs
        device (str): Device to train on

    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    for epoch in range(num_epochs):
        query_encoder.train()
        document_encoder.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Move data to device (query, positive, negative)
            queries = batch["query"]
            positive_docs = batch["positive_doc"]
            negative_docs = batch["negative_doc"]

            # Generate embeddings for queries and documents
            query_embeddings = query_encoder(queries)
            positive_embeddings = document_encoder(positive_docs)
            negative_embeddings = document_encoder(negative_docs)

            # Compute the contrastive loss
            loss, _, _ = contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
