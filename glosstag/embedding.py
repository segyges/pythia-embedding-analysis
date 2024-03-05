import torch
import torch.nn as nn

class GPTNeoXRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )
        
if __name__ == "__main__":
    # sample test case
    pos = torch.tensor([[1], [2]])

    expected_encoding = (
        torch.tensor([
            [1.0, 1.0],
            [0.5403, 0.5403],
            [-0.4161, -0.4161],
            [-0.9900, -0.9900]
    ]), torch.tensor([
            [0.0, 0.0],
            [0.8415, 0.8415],
            [0.9093, 0.9093],
            [0.1411, 0.1411]
    ]))

    actual_encoding = GPTNeoXRotaryEmbedding(2)(pos, 4)

    for e, a in zip(expected_encoding, actual_encoding):
        assert torch.allclose(e, a, atol=1e-4), "ROPEs don't match"
        