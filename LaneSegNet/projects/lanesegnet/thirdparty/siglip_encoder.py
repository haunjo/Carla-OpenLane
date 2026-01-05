"""
SigLIP Text Encoder for Text-Guided Topology Reasoning

Provides frozen SigLIP text embeddings for domain-invariant semantic guidance.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


TEXT_TEMPLATES = [
    "straight lane on regular road",                  # 0
    "straight lane connecting to intersection",       # 1
    "straight lane within intersection area",         # 2
    "left-turning lane on regular road",              # 3
    "left-turning lane connecting to intersection",   # 4
    "left-turning lane within intersection area",     # 5
    "right-turning lane on regular road",             # 6
    "right-turning lane connecting to intersection",  # 7
    "right-turning lane within intersection area"     # 8
]


class SigLIPTextEncoder(nn.Module):
    """
    SigLIP text encoder wrapper for text embedding extraction.

    Features:
    - Frozen pretrained weights (domain-invariant)
    - 768-dim text embeddings (SigLIP-Large)
    - Batch encoding support

    Args:
        model_name: HuggingFace model name (default: "google/siglip-large-patch16-384")
        freeze: Whether to freeze encoder weights (default: True)
    """

    def __init__(self, model_name="google/siglip-so400m-patch14-384", freeze=True):
        super().__init__()

        # Load pretrained SigLIP model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze weights for domain-invariant semantic
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.embed_dim = self.model.config.text_config.hidden_size  # 768 for large

    def forward(self, texts):
        """
        Encode text strings to embeddings.

        Args:
            texts: List of text strings

        Returns:
            embeddings: [N, 768] text embeddings
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Extract text embeddings
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        return outputs

    @torch.no_grad()
    def encode_templates(self, templates=None):
        """
        Encode predefined text templates.

        Args:
            templates: List of text templates (default: 9 lane templates)

        Returns:
            embeddings: [9, 768] template embeddings
        """
        if templates is None:
            templates = TEXT_TEMPLATES

        return self.forward(templates)


def load_siglip_embeddings(templates=None, model_name="google/siglip-so400m-patch14-384", device='cuda'):
    """
    Convenience function to load frozen SigLIP text embeddings.

    Args:
        templates: Text templates to encode (default: 9 lane templates)
        model_name: SigLIP model variant
        device: Device to load embeddings on

    Returns:
        embeddings: [9, 768] frozen text embeddings
    """
    if templates is None:
        templates = TEXT_TEMPLATES

    encoder = SigLIPTextEncoder(model_name=model_name, freeze=True)
    encoder = encoder.to(device)

    with torch.no_grad():
        embeddings = encoder.encode_templates(templates)

    return embeddings


if __name__ == "__main__":
    # Test SigLIP encoder
    print("Testing SigLIP Text Encoder...")

    encoder = SigLIPTextEncoder()
    print(f"Embedding dim: {encoder.embed_dim}")

    # Test template encoding
    embeddings = encoder.encode_templates()
    print(f"Template embeddings shape: {embeddings.shape}")

    # Verify frozen
    print(f"Frozen: {not any(p.requires_grad for p in encoder.parameters())}")

    # Test similarity
    import torch.nn.functional as F
    embeddings_norm = F.normalize(embeddings, dim=-1)
    similarity = embeddings_norm @ embeddings_norm.T
    print(f"\nSimilarity matrix:\n{similarity}")
    print(f"Mean similarity (off-diagonal): {(similarity.sum() - 9) / (81 - 9):.4f}")
