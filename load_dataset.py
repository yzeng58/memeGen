def load_dataset(
    dataset_name: str,
    binary_classification: bool = False,
):
    if dataset_name == 'memotion':
        from load_datasets.load_memotion import load_memotion
        return load_memotion(binary_classification=binary_classification)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")