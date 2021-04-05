import math
import torch
import torch.nn.functional as F


class Mechanism:
    def __init__(self, eps, input_range, **kwargs):
        self.eps = eps
        self.alpha, self.beta = input_range

    def __call__(self, x):
        raise NotImplementedError


class Piecewise(Mechanism):
    def __call__(self, x):
        # normalize x between -1,1
        t = (x - self.alpha) / (self.beta - self.alpha)
        t = 2 * t - 1

        # piecewise mechanism's variables
        P = (math.exp(self.eps) - math.exp(self.eps / 2)) / (2 * math.exp(self.eps / 2) + 2)
        C = (math.exp(self.eps / 2) + 1) / (math.exp(self.eps / 2) - 1)
        L = t * (C + 1) / 2 - (C - 1) / 2
        R = L + C - 1

        # thresholds for random sampling
        threshold_left = P * (L + C) / math.exp(self.eps)
        threshold_right = threshold_left + P * (R - L)

        # masks for piecewise random sampling
        x = torch.rand_like(t)
        mask_left = x < threshold_left
        mask_middle = (threshold_left < x) & (x < threshold_right)
        mask_right = threshold_right < x

        # random sampling
        t = mask_left * (torch.rand_like(t) * (L + C) - C)
        t += mask_middle * (torch.rand_like(t) * (R - L) + L)
        t += mask_right * (torch.rand_like(t) * (C - R) + R)

        # unbias data
        x_prime = (self.beta - self.alpha) * (t + 1) / 2 + self.alpha
        return x_prime


class MultiDimPiecewise(Piecewise):
    def __call__(self, x):
        n, d = x.size()
        k = int(max(1, min(d, math.floor(self.eps / 2.5))))
        sample = torch.rand_like(x).topk(k, dim=1).indices
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, sample, True)
        self.eps /= k
        y = super().__call__(x)
        z = mask * y * d / k
        return z


class RandomizedResponseTopK:
    def __init__(self, eps, k):
        """
        Args:
            eps: privacy budget parameter epsilon
            k: limits choices to top-k items having the highest probabilities
        """
        self.eps = eps
        self.k = k

    def __call__(self, y, p):
        """
        Perturbs input vector y with prior probabilities indicated by p
        Args:
            y: tensor with shape (N,1) with one private label per row
            p: tensor with shape (N,D) with one row per label incdicating the prior distribution
                over D possible outcomes

        Returns:
            perturbed labels
        """
        # add a small random noise to break ties randomly
        p += torch.randn_like(p) / 1e4

        # select top-k items
        kth_max = p.sort(dim=1, descending=True).values.gather(1, self.k - 1)
        included = p >= kth_max

        # apply randomized response on top-k items
        p_incorrect = 1.0 / (math.exp(self.eps) + self.k - 1)
        pr = torch.ones_like(p) * p_incorrect
        y = y.unsqueeze(dim=1) if len(y.size()) == 1 else y
        pr.scatter_(1, y, math.exp(self.eps) * p_incorrect)
        pr.mul_(included)
        pr.mul_(1.0 / pr.sum(dim=1).unsqueeze(1))
        return torch.multinomial(pr, num_samples=1).squeeze()


class RandomizedResponseWithPrior:
    def __init__(self, eps):
        """
        Args:
            eps: privacy budget parameter epsilon
        """
        self.eps = eps

    def __call__(self, y, p):
        """
        Perturbs input vector y with prior probabilities indicated by p
        Args:
            y: tensor with shape (N,1) with one private label per row
            p: tensor with shape (N,D) with one row per label incdicating the prior distribution
                over D possible outcomes

        Returns:
            perturbed labels
        """

        # add a small random noise to break ties randomly
        p_noisy = p + (torch.randn_like(p) / 1e4)

        # select best value of k for RRTop-K
        n, d = p.size()
        k = torch.tensor([range(d)] * n, dtype=float, device=y.device) + 1
        p_noisy = p_noisy.sort(dim=1, descending=True).values
        w = math.exp(self.eps) / (math.exp(self.eps) + k - 1)
        w.mul_(p_noisy.cumsum(dim=1))
        k = w.argmax(dim=1, keepdim=True) + 1
        return RandomizedResponseTopK(eps=self.eps, k=k)(y, p)


class OptimizedUnaryEncoding:
    def __init__(self, eps, d):
        self.d = d
        self.p = 0.5
        self.q = 1 / (math.exp(eps) + 1)

    def __call__(self, y):
        pr = y * self.p + (1 - y) * self.q
        return torch.bernoulli(pr)

    def estimate(self, b):
        n = b.size(0)
        return (b.sum(dim=0) / n - self.q) / (self.p - self.q)
