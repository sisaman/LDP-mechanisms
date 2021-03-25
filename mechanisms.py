import math
import torch


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

