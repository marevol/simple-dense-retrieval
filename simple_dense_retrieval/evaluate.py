import logging

import torch
import torch.nn.functional as F

from simple_dense_retrieval.train import contrastive_loss


def evaluate_dense_retrieval(query_encoder, document_encoder, dataloader, device="cpu"):
    """
    Evaluate the Dense Retrieval model on the validation dataset.

    Args:
        query_encoder (QueryEncoder): Query encoder model.
        document_encoder (DocumentEncoder): Document encoder model.
        dataloader (DataLoader): Validation data loader.
        device (str): Device to evaluate on.

    Returns:
        float: Average loss
        float: Accuracy
    """
    logger = logging.getLogger(__name__)
    query_encoder.eval()
    document_encoder.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get query and document data
            queries = batch["query"]
            positive_docs = batch["positive_doc"]
            negative_docs = batch["negative_doc"]

            # Generate embeddings
            query_embeddings = query_encoder(queries)
            positive_embeddings = document_encoder(positive_docs)
            negative_embeddings = document_encoder(negative_docs)

            # Calculate loss
            loss, pos_sim, neg_sim = contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss.item()

            # Calculate accuracy: Predict 1 if pos_scores > neg_scores, else 0
            preds = (pos_sim > neg_sim).float()  # 1 for positive, 0 for negative

            # True labels: 1 for positive, 0 for negative
            targets = torch.ones_like(pos_sim).to(device)  # All positive samples are target 1
            total += targets.size(0)

            # Compare predictions with actual labels
            correct += (preds == targets).sum().item()

            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        logger.info(f"Evaluation - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
