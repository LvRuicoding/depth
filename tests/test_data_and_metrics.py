import torch

from occany_depth_min.data import (
    DOMAIN_NAMES,
    DomainBalancedDistributedSampler,
    UnifiedSixDataset,
)
from occany_depth_min.metrics import UnifiedDepthMetricAccumulator


class _Sized(torch.utils.data.Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


def test_exact_balanced_sampling_shape() -> None:
    lengths = (3659, 12650, 26000, 795, 5285, 48259)
    dataset = UnifiedSixDataset(
        {name: _Sized(length) for name, length in zip(DOMAIN_NAMES, lengths)}
    )
    sampler = DomainBalancedDistributedSampler(
        dataset, coverage_epochs=10, num_replicas=4, rank=0, seed=0
    )
    assert sampler.samples_per_domain == 4826
    assert sampler.logical_samples_per_epoch == 28956
    assert len(sampler) == 7239


def test_metric_identity() -> None:
    depth = torch.linspace(0.5, 4.0, 64).reshape(8, 8)
    result = UnifiedDepthMetricAccumulator("7scenes").update_image(
        depth, depth, valid_mask=torch.ones_like(depth, dtype=torch.bool)
    ).compute()
    metrics = result["per_image_macro"]["all_valid"]
    assert metrics["abs_rel"] == 0.0
    assert metrics["delta1"] == 1.0
