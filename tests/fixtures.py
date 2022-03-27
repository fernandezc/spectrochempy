import pytest

from spectrochempy import pathclean, preferences as prefs, NDDataset

datadir = pathclean(prefs.datadir)
dataset = NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")


@pytest.fixture(scope="session")
def IR_dataset_2D():
    nd = dataset.copy()
    nd.name = "IR_2D"
    return nd


@pytest.fixture(scope="session")
def IR_dataset_1D():
    nd = dataset[0].squeeze().copy()
    nd.name = "IR_1D"
    return nd


@pytest.fixture(scope="session")
def NMR_dataset_1D():
    path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d" / "1" / "fid"
    dataset = NDDataset.read_topspin(path, remove_digital_filter=True, name="NMR_1D")
    return dataset.copy()


@pytest.fixture(scope="session")
def NMR_dataset_2D():
    path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d" / "1" / "ser"
    dataset = NDDataset.read_topspin(
        path, expno=1, remove_digital_filter=True, name="NMR_2D"
    )
    return dataset.copy()
