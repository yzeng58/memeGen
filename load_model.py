def load_model(
    model_name: str,
    api_key: str = 'yz',
):
    if 'llama' in model_name.lower():
        from load_models.load_llama import load_llama, call_llama
        model = load_llama(model_name, api_key)
        return lambda *args, **kwargs: call_llama(model, *args, **kwargs)
    elif 'claude' in model_name.lower():
        from load_models.load_claude import load_claude, call_claude
        model = load_claude(model_name, api_key)
        return lambda *args, **kwargs: call_claude(model, *args, **kwargs)
    elif 'gpt' in model_name.lower():
        from load_models.load_gpt import load_gpt, call_gpt
        model = load_gpt(model_name, api_key)
        return lambda *args, **kwargs: call_gpt(model, *args, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not found")