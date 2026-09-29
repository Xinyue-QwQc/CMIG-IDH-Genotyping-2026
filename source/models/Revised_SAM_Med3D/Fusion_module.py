import torch.nn as nn
import math
import torch
from einops import rearrange


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads: int):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q.to(self.q_proj.weight.dtype))
        k = self.k_proj(k.to(self.k_proj.weight.dtype))
        v = self.v_proj(v.to(self.v_proj.weight.dtype))
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


# 目前已有参数量4.13M
class AlignImageTextV1(nn.Module):
    def __init__(self, img_channels, text_channels, reduction_ratio=4) -> None:
        """
        img_channels: The input dimension of the img features.
        text_channels: The input dimension of the text features.
        """
        super().__init__()
        hidden_state = img_channels // reduction_ratio

        # For shallow img features
        self.img_proj1 = nn.Sequential(
            nn.LayerNorm(img_channels),
            nn.Linear(img_channels, hidden_state, bias=False),
        )
        self.img_dw1 = nn.Conv3d(hidden_state, hidden_state, kernel_size=3, stride=1, padding=1, groups=hidden_state)

        # For deep img features
        self.img_proj2 = nn.Sequential(
            nn.LayerNorm(img_channels),
            nn.Linear(img_channels, hidden_state, bias=False),
        )
        self.img_dw2 = nn.Conv3d(hidden_state, hidden_state, kernel_size=3, stride=1, padding=1, groups=hidden_state)

        self.text_proj = nn.Linear(text_channels, hidden_state, bias=False)
        self.shift_scale = nn.Linear(text_channels, hidden_state * 2, bias=False)

        self.attn1 = Attention(hidden_state, num_heads=12)
        self.attn2 = Attention(hidden_state, num_heads=12)

        self.concat_proj = nn.Linear(hidden_state * 2, hidden_state, bias=False)

        self.norm = nn.LayerNorm(hidden_state)

    def forward(self, img_shallow, img_deep, text, text_global):
        img_s = self.img_proj1(img_shallow)
        img_s = img_s.permute(0, 4, 1, 2, 3).contiguous()
        img_s = self.img_dw1(img_s)
        img_s = rearrange(img_s, 'b c d h w -> b (d h w) c').contiguous()

        img_d = self.img_proj2(img_deep)
        img_d = img_d.permute(0, 4, 1, 2, 3).contiguous()
        img_d = self.img_dw2(img_d)
        img_d = rearrange(img_d, 'b c d h w -> b (d h w) c').contiguous()

        text = self.text_proj(text)

        shift_scale = self.shift_scale(text_global)
        shift, scale = shift_scale.chunk(2, dim=-1)

        img_s2text = self.attn1(img_s, text, text)
        img_d2text = self.attn2(img_d, text, text)

        img_f = img_s2text + img_d2text

        img_f_mean = img_f.mean(dim=1)
        img_f_max = img_f.max(dim=1).values

        img_f = torch.cat([img_f_mean, img_f_max], dim=-1)

        img_f = self.concat_proj(img_f)
        img_f = self.norm(img_f)

        img_f = img_f * (1 + scale) + shift

        return img_f

class AlignImageText(nn.Module):
    def __init__(self, img_channels, text_channels, reduction_ratio=4) -> None:
        """
        img_channels: The input dimension of the img features.
        text_channels: The input dimension of the text features.
        """
        super().__init__()
        hidden_state = img_channels // reduction_ratio

        self.img_proj1 = nn.Linear(img_channels,hidden_state)

        # For shallow img features
       
        self.img_dw1 = nn.Sequential(
            nn.BatchNorm3d(img_channels),
            nn.Conv3d(img_channels,img_channels,kernel_size=3,padding=1,groups=img_channels)
        )

        # For deep img features
        # self.img_proj2 = nn.Linear(img_channels,hidden_state)
        # self.img_dw2 = nn.Sequential(
        #     nn.BatchNorm3d(img_channels),
        #     nn.Conv3d(img_channels,img_channels,kernel_size=3,padding=1,groups=img_channels)
        # )

        self.text_proj = nn.Linear(text_channels, hidden_state, bias=False)
        self.shift_scale = nn.Linear(text_channels, hidden_state * 2, bias=False)

        self.attn1 = Attention(hidden_state, num_heads=12)
        self.attn2 = Attention(hidden_state, num_heads=12)

        self.concat_proj = nn.Linear(hidden_state * 2, hidden_state, bias=False)

        self.norm = nn.LayerNorm(hidden_state)

    def forward(self, img_s, img_d, text, text_global):
        # 
        img_s = img_s.permute(0, 4, 1, 2, 3).contiguous()
        img_s = self.img_dw1(img_s)
        img_s = rearrange(img_s, 'b c d h w -> b (d h w) c').contiguous()
        img_s = self.img_proj1(img_s)

        # 
        img_d = img_d.permute(0, 4, 1, 2, 3).contiguous()
        img_d = self.img_dw1(img_d)
        img_d = rearrange(img_d, 'b c d h w -> b (d h w) c').contiguous()
        img_d = self.img_proj1(img_d)

        text = self.text_proj(text)

        shift_scale = self.shift_scale(text_global)
        shift, scale = shift_scale.chunk(2, dim=-1)

        img_s2text = self.attn1(img_s, text, text)
        img_d2text = self.attn2(img_d, text, text)

        img_f = img_s2text + img_d2text

        img_f_mean = img_f.mean(dim=1)
        img_f_max = img_f.max(dim=1).values

        img_f = torch.cat([img_f_mean, img_f_max], dim=-1)

        img_f = self.concat_proj(img_f)
        img_f = self.norm(img_f)

        img_f = img_f * (1 + scale) + shift

        return img_f
    
if __name__ == "__main__":
    # 计算参数量1
    # model = AlignImageText(768,1024)
    model = AlignImageText(768, 1024,reduction_ratio=2)
    # 测试代码是否可行
    img_shallow = torch.randn(2, 8, 8, 8, 768)
    img_deep = torch.randn(2, 8, 8, 8, 768)
    text = torch.randn(2, 14, 1024)
    text_global = torch.randn(2, 1024)
    out = model(img_shallow, img_deep, text, text_global)
    print(out.shape)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(params)
