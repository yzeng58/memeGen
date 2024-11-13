def process_score(score):
    try:
        return int(score)
    except:
        return -1


def get_score(
    q, 
    meme_path,
    call_model,
    output_control = "Please answer the question with a number without any other words.",
    example = False,
    max_intermediate_tokens = 300,
    max_new_tokens = 1,
    description = '',
    context = '',
):
    output_1 = call_model(
        f"{q['question']} {q['rating']}" + (" " + q['example'] if example else ''),
        [meme_path],
        max_new_tokens=max_intermediate_tokens,
        save_history=True,
        description = description,
        context = context,
    )

    output_2 = call_model(
        q['rating'] + output_control,
        [],
        max_new_tokens = max_new_tokens,
        history = output_1['history'],
        save_history = True,
        description = description,
        context = context,
    )

    output_dict = {
        'score': process_score(output_2['output']),
        'analysis': output_1['output'] + output_2['output'],
    }

    return output_dict