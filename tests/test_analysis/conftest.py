import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close()
