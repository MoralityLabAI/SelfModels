class SpectralOutcomeProjector(nn.Module):
    """
    eta(O): frequency-calibrated sketch of logits.
    - extracts top-k logits
    - projects to d_eta complex bins
    - FFT + learned complex weights
    - returns eta as real vector of size 2*d_eta
    """
    def __init__(self, vocab_size: int, d_eta: int, d_obs: int, topk: int = 64):
        super().__init__()
        self.d_eta = d_eta
        self.topk = topk

        # maps (2*topk) -> (2*d_eta) so we can view_as_complex
        self.to_complex_bins = nn.Sequential(
            nn.Linear(2 * topk, 256),
            nn.GELU(),
            nn.Linear(256, 2 * d_eta),
        )

        # complex frequency calibration
        self.freq_weights = nn.Parameter(torch.view_as_complex(torch.randn(d_eta, 2)))
        self.token_obs_emb = nn.Embedding(vocab_size, d_obs)

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(self.topk, logits.shape[-1])
        vals, idx = torch.topk(logits, k=k)
        probs = torch.softmax(vals, dim=-1)

        # feature is (vals, probs) padded to topk
        if k < self.topk:
            pad = self.topk - k
            vals = torch.cat([vals, vals.new_zeros(pad)], dim=0)
            probs = torch.cat([probs, probs.new_zeros(pad)], dim=0)

        feat = torch.cat([vals, probs], dim=0)                 # (2*topk,)
        bins_ri = self.to_complex_bins(feat)                   # (2*d_eta,)
        bins_c = torch.view_as_complex(bins_ri.view(self.d_eta, 2))  # (d_eta,) complex

        spectrum = torch.fft.fft(bins_c) * self.freq_weights   # (d_eta,) complex
        eta = torch.view_as_real(spectrum).flatten()           # (2*d_eta,)

        # expected obs embedding under top-k probs
        # use original k values, not padded
        vals0, idx0 = torch.topk(logits, k=min(self.topk, logits.shape[-1]))
        probs0 = torch.softmax(vals0, dim=-1)
        exp_obs = (probs0.unsqueeze(-1) * self.token_obs_emb(idx0)).sum(dim=0)

        return eta, exp_obs
