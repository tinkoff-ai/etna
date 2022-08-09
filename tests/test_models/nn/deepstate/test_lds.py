import pytest
import torch
import torch.testing

from etna.models.nn import LDS


@pytest.fixture
def lds(batch_size=2, seq_length=10, latent_dim=2):
    lds = LDS(
        emission_coeff=torch.rand(size=(batch_size, seq_length, latent_dim)).float(),
        transition_coeff=torch.rand(size=(latent_dim, latent_dim)).float(),
        innovation_coeff=torch.rand(size=(batch_size, seq_length, latent_dim)).float(),
        noise_std=torch.rand(size=(batch_size, seq_length, 1)).float(),
        prior_mean=torch.rand(size=(batch_size, latent_dim)).float(),
        prior_std=torch.rand(size=(batch_size, latent_dim)).float(),
        offset=torch.rand(size=(batch_size, seq_length, 1)).float(),
        seq_length=seq_length,
        latent_dim=latent_dim,
    )
    return lds


def test_sample_initials_shape(lds, n_samples=3):
    l_0_expected_shape = (n_samples, lds.batch_size, lds.latent_dim)
    eps_latent_expected_shape = (n_samples, lds.batch_size, lds.seq_length, lds.latent_dim)
    eps_observation_expected_shape = (n_samples, lds.batch_size, lds.seq_length, 1)

    l_0, eps_latent, eps_observation = lds._sample_initials(n_samples=n_samples)
    assert l_0.shape == l_0_expected_shape
    assert eps_latent.shape == eps_latent_expected_shape
    assert eps_observation.shape == eps_observation_expected_shape


def test_sample_shape(lds, n_samples=3):
    expected_shape = (n_samples, lds.batch_size, lds.seq_length, 1)

    samples = lds.sample(n_samples=n_samples)
    assert samples.shape == expected_shape
