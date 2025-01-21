import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from rate_meme.score_meme_v1 import score_meme_based_on_theory_v1
from rate_meme.score_meme_v2 import score_meme_based_on_theory_v2
from rate_meme.score_meme_v3 import score_meme_based_on_theory_v3
from rate_meme.score_meme_v4 import score_meme_based_on_theory_v4
from rate_meme.score_meme_v5 import score_meme_based_on_theory_v5
from rate_meme.score_meme_v6 import score_meme_based_on_theory_v6

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
    system_prompt_name = 'default',
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
            system_prompt_name = system_prompt_name,
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
            system_prompt_name = system_prompt_name,
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
            system_prompt_name = system_prompt_name,
        )
    elif version == 'v4':
        return score_meme_based_on_theory_v4(
            meme_path = meme_path,
            call_model = call_model,
            result_dir = result_dir,
            max_new_tokens = max_intermediate_tokens,
            example = example,
            description = description,
            context = context,
            overwrite = overwrite,
            system_prompt_name = system_prompt_name,
        )
    elif version == 'v5':
        return score_meme_based_on_theory_v5(
            meme_path = meme_path,
            call_model = call_model,
            result_dir = result_dir,
            max_new_tokens = max_new_tokens,
            example = example,
            description = description,
            context = context,
            overwrite = overwrite,
            system_prompt_name = system_prompt_name,
        )
    elif version == 'v6':
        return score_meme_based_on_theory_v6(
            meme_path = meme_path,
            call_model = call_model,
            result_dir = result_dir,
            max_new_tokens = max_intermediate_tokens,
            example = example,
            description = description,
            context = context,
            overwrite = overwrite,
            system_prompt_name = system_prompt_name,
        )
    else:
        raise ValueError(f"Version {version} not supported!")