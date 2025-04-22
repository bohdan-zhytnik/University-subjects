import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ImagePatchEmbeding# #{
class ImagePatchEmbeding(nn.Module):
    def __init__(self, img_size: int=32, patch_size: int=4, in_chans: int=3, embed_dim: int=48):
        """
        ImagePatchEmbeding: This class is used to convert the image patches into a higher dimension space, which is the embedding
        Args:
            img_size (int): The size of the image.
            patch_size (int): The desired size of the patch.
            in_chans (int): The number of input channels.
            embed_dim (int): The dimension of the embedding.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Divide image into patches and embed them
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size, p2=patch_size),
            nn.Linear(in_chans * patch_size * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ImagePatchEmbeding class.
        Args:
            x (torch.Tensor): The input image tensor of shape (B, C, W, W).
        Returns:
            torch.Tensor: The embedded image tensor of shape (B, num_patches, embed_dim).
        """
        x = self.to_patch_embedding(x)
        return x
# # #}

# EmbedingCreation# #{
class EmbedingCreation(nn.Module):
    """
    EmbedingCreation: This class is used to add the positional encoding to the patches and add the class token to the patches.
    Args:
        num_patches (int): The number of patches in the image.
        embed_dim (int): The dimension of the embedding.
        dropout (float): The dropout probability.
    """
    def __init__(self, img_size: int=32, num_patches: int=0, patch_size: int=4, embed_dim: int=48, dropout: float=0.0):
        super().__init__()
        self.num_patches = num_patches
        self.patch_embed = ImagePatchEmbeding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EmbedingCreation class.
        Args:
            x (torch.Tensor): The input image tensor of shape (B, C, W, W).
        Returns:
            torch.Tensor: The embedded image tensor of shape (B, num_patches + 1, embed_dim).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x
# # #}

# TransformerHead# #{
class TransformerHead(nn.Module):
    """
    TransformerHead: This class is used to implement the transformer head.
    Args:
        embed_dim (int): The dimension of the embedding.
        head_dim (int): The dimension of the head.
        qkv_bias (bool): If True, then the qkv linear layers will have bias.
        dropout (float): The dropout probability.
    """
    def __init__(self, embed_dim: int=48, head_dim: int=65, qkv_bias: bool=True, dropout: float=0.0):
        super().__init__()
        if head_dim is None:
            head_dim = embed_dim
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        self.qkv = nn.Linear(embed_dim, 3 * head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(head_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the TransformerHead class.
        Args:
            x (torch.Tensor): The input tensor of shape (B, num_patches + 1, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (B, num_patches + 1, embed_dim). 
            torch.Tensor: The probabilities matrix (right after softmax) output tensor of shape (B, num_patches + 1, num_patches + 1).
        """
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        sa_prob = attn
        attn = self.attn_drop(attn)
        sa_out = attn @ v
        sa_out = self.proj(sa_out)
        sa_out = self.proj_drop(sa_out)
        return (sa_out, sa_prob)
# # #} 

# TransformerMultiHeadAttention# #{
class TransformerMultiHeadAttention(nn.Module):
    """
    TransformerMultiHeadAttention: This class is used to implement the multi-head attention mechanism.
    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of heads in the multi-head attention.
        qkv_bias (bool): If True, then the qkv linear layers will have bias.
        dropout (float): The dropout probability.
    """
    def __init__(self, embed_dim: int=768, num_heads: int=12, qkv_bias: bool=True, dropout: float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.heads = nn.ModuleList([
            TransformerHead(embed_dim=self.embed_dim, head_dim=self.head_dim, qkv_bias=self.qkv_bias, dropout=self.dropout)
            for _ in range(self.num_heads)
        ])
        # print("self.num_heads * self.head_dim",self.num_heads * self.head_dim)
        self.proj = nn.Linear(self.num_heads * self.num_heads * self.head_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the TransformerMultiHeadAttention class.
        Args:
            x (torch.Tensor): The input tensor of shape (B, num_patches + 1, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (B, num_patches + 1, embed_dim).
            torch.Tensor: The attention probabilities taken from the output of the TransformerHead. 
        """
        B, T, C = x.shape
        heads_output = []
        sa_probs_list = []
        for head in self.heads:
            out, prob = head(x)
            heads_output.append(out)
            sa_probs_list.append(prob)
        # print("heads_output",heads_output.shape)
        sa_out = torch.cat(heads_output, dim=-1)
        # print("sa_out before proj",sa_out.shape)
        sa_out = self.proj(sa_out)
        # print("sa_out after proj",sa_out.shape)
        sa_out = self.dropout(sa_out)

        sa_probs = torch.stack(sa_probs_list, dim=0)
        sa_probs = sa_probs.permute(1,0,2,3)
        return sa_out, sa_probs
# # #}

# FeedForward# #{
class FeedForward(nn.Module):
    """
    FeedForward: This class is used to implement the feed forward neural network in the transformer block.
    Args:
        embed_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden layer.
        dropout (float): The dropout probability.
    """
    def __init__(self, embed_dim: int=768, hidden_dim: int=3072, dropout: float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward class.
        Args:
            x (torch.Tensor): The input tensor of shape (B, num_patches + 1, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (B, num_patches + 1, embed_dim).
        """
        return self.ffwd(x)
# # #}

# Block# #{
class Block(nn.Module):
    """
    Block: This class is used to implement a single transformer block.
    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of heads in the multi-head attention.
        qkv_bias (bool): If True, then the qkv linear layers will have bias.
        dropout (float): The dropout probability.
    """

    def __init__(self, embed_dim: int=768, num_heads: int=12, qkv_bias: bool=True, dropout: float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.attn = TransformerMultiHeadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, dropout=self.dropout)
        self.ffw = FeedForward(embed_dim=self.embed_dim, dropout=self.dropout)
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the Transformer Block class.
        Args:
            x (torch.Tensor): The input tensor of shape (B, num_patches + 1, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (B, num_patches + 1, embed_dim).
            torch.Tensor: The attention probabilities taken from the output of the TransformerMultiHeadAttention of shape (B, num_heads, num_patches + 1, num_patches + 1).
        """
        x_res = x
        x = self.ln1(x)
        sa_out, sa_probs = self.attn(x)
        x = x_res + sa_out
        x_res = x
        x = self.ln2(x)
        x = x_res + self.ffw(x)
        return x, sa_probs
# # #}

# TransformerEncoder# #{
class TransformerEncoder(nn.Module):
    """
    TransformerEncoder: This class is used to implement the transformer encoder.
    Args:
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of heads in the multi-head attention.
        qkv_bias (bool): If True, then the qkv linear layers will have bias.
        dropout (float): The dropout probability.
        num_layers (int): The number of transformer blocks.
    """
    def __init__(self, embed_dim: int=768, num_heads: int=12, qkv_bias: bool=True, dropout: float=0.0, num_layers: int=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.blocks = nn.ModuleList([Block(embed_dim=self.embed_dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, dropout=self.dropout) for _ in range(self.num_layers)])

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the TransformerEncoder class.
        Args:
            x (torch.Tensor): The input tensor of shape (B, num_patches + 1, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (B, num_patches + 1, embed_dim).
            torch.Tensor: 
        """
        sa_probs = []
        for blk in self.blocks:
            x, p = blk(x)
            sa_probs.append(p)
        sa_probs = torch.stack(sa_probs, dim=0) # (num_layers, B, num_heads, T, T)
        return x, sa_probs
# # #}

# VisionTransformerClassifier# #{
class VisionTransformerClassifier(nn.Module):
    """
    VisionTransformer: This class is used to implement the Vision Transformer for the image classification task.
    Args:
        img_size (int): The size of the image.
        embed_dim (int): The dimension of the embedding.
        num_heads (int): The number of heads in the multi-head attention.
        qkv_bias (bool): If True, then the qkv linear layers will have bias.
        dropout (float): The dropout probability.
        num_layers (int): The number of transformer blocks.
        num_classes (int): The number of classes in the dataset.
    """
    def __init__(self, img_size: int=32, embed_dim: int=48, num_heads: int=12, patch_size: int=4, qkv_bias: bool=True, dropout: float=0.0, num_layers: int=4, num_classes: int=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size)**2
        self.embedding = EmbedingCreation(img_size=self.img_size, num_patches=self.num_patches, patch_size=self.patch_size, embed_dim=self.embed_dim, dropout=self.dropout)
        self.encoder = TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, dropout=self.dropout, num_layers=self.num_layers)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.apply(self.init_weights) 

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the VisionTransformer class.
        Args:
            x (torch.Tensor): The input tensor of shape (B, C, W, W).
        Returns:
            torch.Tensor: The output tensor of shape (B, num_classes).
            torch.Tensor: 
        """
        x = self.embedding(x)
        x, sa_probs = self.encoder(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x, sa_probs

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, EmbedingCreation):
            module.pos_embedding.data = nn.init.trunc_normal_(
                module.pos_embedding.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.pos_embedding.dtype)

            module.class_token.data = nn.init.trunc_normal_(
                module.class_token.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.class_token.dtype)

# # #}

if __name__ == "__main__":
    model = VisionTransformerClassifier()
    x = torch.randn(1, 3, 32, 32)
    y, sa_probs = model(x)
