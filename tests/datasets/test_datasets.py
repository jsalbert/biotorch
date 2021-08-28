from biotorch.datasets.selector import DatasetSelector


def test_datasets_implemented(datasets_available):
    for dataset_name in datasets_available:
        assert DatasetSelector(dataset_name).get_dataset()
