import torch
from torch import nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.model = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, bias=False)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        def forward(
                self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                is_causal: bool = False,
        ) -> tuple[Tensor, Optional[Tensor]]:

        attn_output, attn_output_weights = F.multi_head_attention_forward(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.bias_k,
                        self.bias_v,
                        self.add_zero_attn,
                        self.dropout,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        training=self.training,
                        key_padding_mask=key_padding_mask,
                        need_weights=need_weights,
                        attn_mask=attn_mask,
                        average_attn_weights=average_attn_weights,
                        is_causal=is_causal,
                    )
        """
        # attn_output, attn_output_weights = F.multi_head_attention_forward(
        #     use_separate_proj_weight=False,
        #     query=Q,
        #     key=K,
        #     value=V,
        #     embed_dim_to_check=self.embed_dim,
        #     num_heads=self.n_heads,
        #     in_proj_weight=None,
        #     in_proj_bias=None,
        #     bias_k=None,
        #     bias_v=None,
        #     out_proj_weight=None,
        #     out_proj_bias=None,
        #     add_zero_attn=False,
        #     dropout_p=0.1)

        attn_output, attn_output_weights = self.model(Q, K, V)
        return attn_output, attn_output_weights


def main():
    device = torch.device("mps")

    n_heads = 1
    d_model = 512  # d_model
    seq_len = 128  # n

    X = torch.randn(seq_len, d_model)  # X (n, d_model)

    # define weights
    WQ = torch.randn(d_model, d_model)
    WK = torch.randn(d_model, d_model)
    WV = torch.randn(d_model, d_model)

    # work on device
    X = X.to(device)
    WQ, WK, WV = WQ.to(device), WK.to(device), WV.to(device)
    Q = torch.mm(X, WQ)
    K = torch.mm(X, WK)
    V = torch.mm(X, WV)
    model = MHA(d_model, n_heads).to(device)
    model.train()

    mha_attn = model(Q, K, V)
    print(mha_attn[0].shape, mha_attn[1].shape)


if __name__ == "__main__":
    main()
