def load_dataset(
    dataset_name: str,
    description: str = '',
    binary_classification: bool = False,
):
    if dataset_name == 'memotion':
        from load_datasets.load_memotion import load_memotion
        if description: raise ValueError('Memotion dataset does not support description')
        return load_memotion(binary_classification=binary_classification)
    elif dataset_name == 'ours_v2':
        from load_datasets.load_ours_v2 import load_ours_v2
        return load_ours_v2(description=description)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    