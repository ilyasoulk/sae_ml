loss_fns = {
    "l1": lambda x, x_dec, features, **kwargs: compute_l1_sae_loss(
        x, x_dec, features, kwargs.get("l1_coeff", 1e-3)
    ),
    "topk": lambda x, x_dec, features, **kwargs: compute_topk_sae_loss(x, x_dec),
}


def sae_loss(x, x_dec, features, loss_type, **kwargs):
    if loss_type not in loss_fns:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Available: {list(loss_fns.keys())}"
        )

    return loss_fns[loss_type](x, x_dec, features, **kwargs)


def compute_l1_sae_loss(x, x_dec, features, l1_coeff):
    mse_loss = (x_dec - x).pow(2).sum(dim=-1).mean()
    l1_loss = features.abs().sum(dim=-1).mean()
    return mse_loss + (l1_coeff * l1_loss)


def compute_topk_sae_loss(x, x_dec):
    return (x_dec - x).pow(2).sum(dim=-1).mean()
