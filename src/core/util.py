import importlib
import torch

def load_and_format(config,data_set):
    dataset = importlib.import_module(f'src.tasks.{config["task"]}.dataset')
    data = dataset.load_data(config[data_set])
    data_formatted = data.map(dataset.format_example, 
                              remove_columns=data.column_names,
                              load_from_cache_file=False)
    return data_formatted

class CausalLMWithLabelsCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # 1) Separate labels so tokenizer.pad doesn't touch them
        labels = [f.pop("labels") for f in features]

        # 2) Pad inputs
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # 3) Pad labels to same sequence length with -100
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) > max_len:
                lab = lab[:max_len]
            else:
                lab = lab + [-100] * (max_len - len(lab))
            padded_labels.append(lab)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

def tokenizer_func(ex, tokenizer):
    # Build texts
    full_text = ex["prompt"] + ex["target"]
    prompt_text = ex["prompt"]

    # Tokenize without padding
    full_enc = tokenizer(full_text, truncation=True, padding=False)
    prompt_enc = tokenizer(prompt_text, truncation=True, padding=False)

    prompt_len = len(prompt_enc['input_ids'])

    # Labels: ignore prompt
    labels = [-100] * prompt_len + full_enc['input_ids'][prompt_len:]
    full_enc['labels'] = labels
    assert len(labels) == len(full_enc['input_ids'])

    return full_enc