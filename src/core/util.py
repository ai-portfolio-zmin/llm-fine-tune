import importlib

def load_and_format(config,data_set):
    dataset = importlib.import_module(f'src.tasks.{config["task"]}.dataset')
    data = dataset.load_data(config[data_set])
    data_formatted = data.map(dataset.format_example, 
                              remove_columns=data.column_names,
                              load_from_cache_file=False)
    return data_formatted


def tokenizer_func(ex, tokenizer, MAX_LEN=512):
    # Build full text and prompt-only text
    full_text = ex["prompt"] + ex["target"]
    prompt_text = ex["prompt"]

    # Tokenize both
    full_enc = tokenizer(full_text, truncation=True, padding=False)
    prompt_enc = tokenizer(prompt_text, truncation=True, padding=False)

    full_ids = full_enc["input_ids"]
    prompt_ids = prompt_enc["input_ids"]
    prompt_len = len(prompt_ids)

    # Build labels: prompt → -100
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # Now pad BOTH to MAX_LEN
    full_ids = full_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]

    # pad end with pad_token + -100
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    while len(full_ids) < MAX_LEN:
        full_ids.append(pad_id)
        labels.append(-100)

    return {
        "input_ids": full_ids,
        "attention_mask": [1 if i != pad_id else 0 for i in full_ids],
        "labels": labels,
    }
