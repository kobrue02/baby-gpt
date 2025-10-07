import torch
from torch import Tensor

def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(
    p: Tensor,
    p_orthonorm: Tensor,
    grad: Tensor,
    momentum: Tensor,
    beta1: float,
    beta2: float,
    lr: float,
    wd: float,
    nesterov: bool=True,
    update_rms_compensate: bool=True,
    update_spectral_compensate: bool=False,
    use_bf16: bool=True,
    decay_after_update: bool=False,
    device: str='cpu'
    ):

    # move to correct device
    p = p.to(device)
    p_orthonorm = p_orthonorm.to(device)
    grad = grad.to(device)
    momentum = momentum.to(device)

    # compensation
    compensation_coeff = (
        (grad.size(-2) / grad.size(-1))**0.5 if update_spectral_compensate else
        max(1, grad.size(-2) / grad.size(-1))**0.5 if update_rms_compensate else
        1
    )
    if wd != 0 and not decay_after_update:
        p.mul_(1-lr*wd)

    # Orthonormalize gradient and update momentum in orthonormal space
    grad_orthonorm = newtonschulz5(grad, use_bf16)
    momentum.lerp_(grad_orthonorm, 1-beta2)

    if nesterov:
        update = grad_orthonorm.lerp(momentum, 1-beta1)
    else:
        update = momentum

    update.mul_(compensation_coeff)

    # Store orthonormalized parameter state
    p_orthonorm.copy_(p.view_as(grad_orthonorm))
    p_orthonorm.add_(update, alpha=-lr)

    # Update original parameter
    p.add_(update.reshape(p.shape), alpha=-lr)

    if wd != 0 and decay_after_update:
        p.mul_(1-lr*wd)

class MuonEnhanced(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/
    """
    def __init__(
            self, 
            params, 
            lr=0.01, 
            weight_decay=0.01,
            beta1=0.95, 
            beta2=0.95, 
            nesterov=True,
            update_rms_compensate=True,
            update_spectral_compensate=False, 
            use_bf16=True,
            decay_after_update=False,
            device='cpu'
        ):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, nesterov=nesterov,
                        update_rms_compensate=update_rms_compensate,
                        update_spectral_compensate=update_spectral_compensate,
                        use_bf16=use_bf16, decay_after_update=decay_after_update)
        super().__init__(params, defaults)
        self.device = device

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["p_orthonorm"] = torch.zeros_like(p)
                muon_update(
                    p=p, 
                    p_orthonorm=state["p_orthonorm"],
                    grad=p.grad,
                    momentum=state["momentum_buffer"], 
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    lr=group["lr"],
                    wd=group["weight_decay"],
                    nesterov=group["nesterov"],
                    update_rms_compensate=group["update_rms_compensate"],
                    update_spectral_compensate=group["update_spectral_compensate"],
                    use_bf16=group["use_bf16"],
                    decay_after_update=group["decay_after_update"]
                )
        return loss