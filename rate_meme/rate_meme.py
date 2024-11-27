import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from rate_meme.score_meme_v1 import score_meme_based_on_theory_v1
from rate_meme.score_meme_v2 import score_meme_based_on_theory_v2
from rate_meme.score_meme_v3 import score_meme_based_on_theory_v3

def score_meme_based_on_theory(
    meme_path,
    call_model,
    result_dir = None,
    max_intermediate_tokens=300,
    max_new_tokens=1,
    example = False,
    description = '',
    context = '',
    overwrite = False,
    version = 'v1',
):
    if version == 'v1':
        return score_meme_based_on_theory_v1(
            meme_path = meme_path,
            call_model = call_model,
            result_dir = result_dir,
            max_intermediate_tokens = max_intermediate_tokens,
            max_new_tokens = max_new_tokens,
            example = example,
            description = description,
            context = context,
            overwrite = overwrite,
        )
    elif version == 'v2':
        return score_meme_based_on_theory_v2(
            meme_path = meme_path,
            call_model = call_model,
            result_dir = result_dir,
            max_intermediate_tokens = max_intermediate_tokens,
            max_new_tokens = max_new_tokens,
            example = example,
            description = description,
            context = context,
            overwrite = overwrite,
        )
    elif version == 'v3':
        return score_meme_based_on_theory_v3(
            meme_path = meme_path,
            call_model = call_model,
            result_dir = result_dir,
            max_new_tokens = max_intermediate_tokens,
            example = example,
            description = description,
            context = context,
            overwrite = overwrite,
        )
    else:
        raise ValueError(f"Version {version} not supported!")