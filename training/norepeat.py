import torch
from transformers import LogitsProcessor


class GenNoRepeat(LogitsProcessor):
    def __init__(self, prompt_len: int, n: int = 4, window: int = 0, penalty: float = 1.0, freq: float = 0.0):
        self.p, self.n, self.window, self.penalty, self.freq = prompt_len, n, window, penalty, freq

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        g = input_ids[:, self.p:]
        T = g.shape[1]
        if self.n > 1 and T >= self.n:
            ctx = g[:, T - (self.n - 1):]
            win = g.unfold(1, self.n, 1)
            hit = (win[:, :, :-1] == ctx[:, None, :]).all(-1)
            hit[:, -1] = False
            b, i = hit.nonzero(as_tuple=True)
            if b.numel():
                scores[b, win[b, i, -1]] = float("-inf")
        if self.freq and T:
            counts = torch.zeros_like(scores).scatter_add_(1, g, torch.ones_like(g, dtype=scores.dtype))
            scores = scores - self.freq * counts
        if self.window and self.penalty != 1.0 and T:
            recent = g[:, -self.window:]
            s = scores.gather(1, recent)
            scores.scatter_(1, recent, torch.where(s > 0, s / self.penalty, s * self.penalty))
        return scores
