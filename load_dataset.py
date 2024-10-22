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
    elif dataset_name == '130k':
        from load_datasets.load_130k import load_130k
        if description: raise ValueError('130k dataset does not support description')
        return load_130k()
    elif dataset_name == "vineeth":
        from load_datasets.load_vineeth import load_vineeth
        return load_vineeth()
    elif dataset_name == "vipul":
        from load_datasets.load_vipul import load_vipul
        return load_vipul()
    elif dataset_name == "nikitricky":
        from load_datasets.load_nikitricky import load_nikitricky
        return load_nikitricky()
    elif dataset_name == "singh":
        from load_datasets.load_singh import load_singh
        return load_singh()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    