def load_dataset(
    dataset_name: str,
    description: str = '',
    binary_classification: bool = False,
    eval_mode: str = 'pairwise',
    train_test_split: bool = False,
    difficulty: str = 'easy',
):
    if dataset_name == 'memotion':
        from load_datasets.load_memotion import load_memotion
        return load_memotion(
            binary_classification=binary_classification,
            description=description
        )
    elif dataset_name == 'ours_v2':
        from load_datasets.load_ours_v2 import load_ours_v2
        return load_ours_v2(description=description)
    elif dataset_name == 'ours_v3':
        from load_datasets.load_ours_v3 import load_ours_v3
        return load_ours_v3(description=description, train_test_split=train_test_split)
    elif dataset_name == 'ours_v4':
        from load_datasets.load_ours_v4 import load_ours_v4
        return load_ours_v4(description=description, train_test_split=train_test_split)
    elif dataset_name == '130k':
        from load_datasets.load_130k import load_130k
        if description: raise ValueError('130k dataset does not support description')
        return load_130k()
    elif dataset_name == "vineeth":
        from load_datasets.load_vineeth import load_vineeth
        if description: raise ValueError('Vineeth dataset does not support description')
        return load_vineeth()
    elif dataset_name == "vipul":
        from load_datasets.load_vipul import load_vipul
        if description: raise ValueError('Vipul dataset does not support description')
        return load_vipul()
    elif dataset_name == "nikitricky":
        from load_datasets.load_nikitricky import load_nikitricky
        if description: raise ValueError('Nikitricky dataset does not support description')
        return load_nikitricky()
    elif dataset_name == "singh":
        from load_datasets.load_singh import load_singh
        if description: raise ValueError('Singh dataset does not support description')
        return load_singh()
    elif dataset_name == "gmor":
        from load_datasets.load_gmor import load_gmor
        if description: raise ValueError('Gmor dataset does not support description')
        return load_gmor()
    elif dataset_name == "tiwari":
        from load_datasets.load_tiwari import load_tiwari
        if description: raise ValueError('Tiwari dataset does not support description')
        return load_tiwari()
    elif dataset_name == "metmeme":
        from load_datasets.load_metmeme import load_metmeme
        if description: raise ValueError('Metmeme dataset does not support description')
        return load_metmeme()
    elif dataset_name == "relca":
        from load_datasets.load_relca import load_relca
        return load_relca(
            description=description, 
            train_test_split=train_test_split, 
            difficulty=difficulty,
        )
    elif dataset_name == "meta_hateful":
        from load_datasets.load_meta_hateful import load_meta_hateful
        return load_meta_hateful(description=description)
    elif dataset_name == "devastator":
        from load_datasets.load_devastator import load_devastator
        return load_devastator(description=description, eval_mode=eval_mode)
    elif dataset_name == "ours_gen_v1":
        from load_datasets.load_ours_gen_v1 import load_ours_gen_v1
        return load_ours_gen_v1()
    elif dataset_name == "isarcasm":
        from load_datasets.load_isarcasm import load_isarcasm
        return load_isarcasm(train_test_split=train_test_split)
    elif dataset_name == "british_complaints":
        from load_datasets.load_british_complaints import load_british_complaints
        return load_british_complaints(train_test_split=train_test_split)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    