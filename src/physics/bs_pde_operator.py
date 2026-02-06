from __future__ import annotations

import torch

def bs_pde_residual(
        *,
        model,
        s: torch.Tensor,
        t: torch.Tensor,
        r: float,
        sigma: float,
) -> torch.Tensor:
    """
        Compute Black-Scholes PDE residual at interior points:
            Vₜ + 0.5 × σ²×S²×V_SS + r×S×V_S - rV

        Arguments:
            s: (n, 1) requires_grad = True
            t: (n, 1) requires_grad = True

        returns:
            residual: (n, 1)

    """
    if s.ndim != 2 or t.ndim != 2 or s.shape[1] != 1 or t.shape[1] != 1:
        raise ValueError("s and t must be shape (n, 1)")

    x = torch.cat([s, t], dim=1) #(n, 2)
    v = model(x)    #(n, 1)

    ones = torch.ones_like(v)

    v_s = torch.autograd.grad(v, s, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    v_ss = torch.autograd.grad(v_s, s, grad_outputs=torch.ones_like(v_s), create_graph=True, retain_graph=True)[0]

    res = v_t + 0.5 * (sigma **2) * (s ** 2) * v_ss + r * s * v_s - r * v
    return res

