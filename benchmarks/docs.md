# Decoder-Only Transformer

```
Input Embeddings (x)
Shape: (B, S, D)
         │
         │ forward pass
         ▼
Causal Mask (tgt_mask)
Shape: (S, S, float32)
         │
         │ applied to self-attention
         ▼
Memory (cross-attention)
Shape: (B, 0, D)
         │
         │ effectively disables cross-attention
         ▼
Decoder Stack
- N layers of:
  ├─ Masked Self-Attention
  │   input: (B, S, D)
  │   mask: (S, S)
  │   output: (B, S, D)
  ├─ Cross-Attention (skipped due to empty memory)
  └─ Feedforward
         │
         ▼
Decoder Output
Shape: (B, S, D)
         │
         │ linear projection to vocabulary
         ▼
LM Head (nn.Linear)
Shape: (B, S, V)
         │
         │ shift for teacher forcing
         ▼
Shifted Logits
Shape: (B, S-1, V)

Shifted Targets (y)
Shape: (B, S-1)

Flatten for Loss:
- Logits: (B*(S-1), V)
- Targets: (B*(S-1),)
Loss Function: CrossEntropyLoss
```

## Legend / Notes
- **B** = batch size  
- **S** = sequence length  
- **D** = embedding dimension (`d_model`)  
- **V** = vocabulary size  
- Memory has zero length → cross-attention disabled  
- Causal mask ensures autoregressive behavior  
- Teacher forcing uses logits[:, :-1, :] and targets[:, 1:]  
- Mask always float32 even when model uses fp16/bf16
