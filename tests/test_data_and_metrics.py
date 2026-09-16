import torch
from torch.utils.data import DistributedSampler

from occany_depth_min.data import (
    DOMAIN_NAMES,
    DomainBalancedDistributedSampler,
    UnifiedSixDataset,
)
from occany_depth_min.metrics import UnifiedDepthMetricAccumulator
from occany_depth_min.train import NATURAL_SAMPLING_CONTRACT, build_training_sampler


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


def test_natural_sampling_is_one_full_unbalanced_pass() -> None:
    lengths = (3659, 12650, 26000, 795, 5285, 48259)
    dataset = UnifiedSixDataset(
        {name: _Sized(length) for name, length in zip(DOMAIN_NAMES, lengths)}
    )
    samplers = [
        build_training_sampler(dataset, sampling="natural", rank=rank, world=4)
        for rank in range(4)
    ]

    assert NATURAL_SAMPLING_CONTRACT.endswith("full_pass_per_epoch_v1")
    assert all(isinstance(sampler, DistributedSampler) for sampler in samplers)
    assert all(len(sampler) == 24162 for sampler in samplers)

    global_indices = [index for sampler in samplers for index in sampler]
    assert len(global_indices) == len(dataset) == 96648
    assert len(set(global_indices)) == len(dataset)
    for name, expected in zip(DOMAIN_NAMES, lengths):
        domain_indices = dataset.domain_indices[name]
        observed = sum(index in domain_indices for index in global_indices)
        assert observed == expected


def test_metric_identity() -> None:
    depth = torch.linspace(0.5, 4.0, 64).reshape(8, 8)
    result = UnifiedDepthMetricAccumulator("7scenes").update_image(
        depth, depth, valid_mask=torch.ones_like(depth, dtype=torch.bool)
    ).compute()
    metrics = result["per_image_macro"]["all_valid"]
    assert metrics["abs_rel"] == 0.0
    assert metrics["delta1"] == 1.0
