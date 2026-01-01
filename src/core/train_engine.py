import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import yaml
from functools import partial
from src.core.util import load_and_format
from src.path_util import get_model_dir
from src.logger_util import get_logger

logger = get_logger('train')

def tokenizer_func(ex, tokenizer, max_length: int = 512):
    full_text = ex["prompt"] + ex["target"]
    prompt_text = ex["prompt"]

    # Ensure we have a pad token (Phi-3 often uses eos as pad)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize to fixed length so every example is max_length
    full_enc = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    prompt_enc = tokenizer(
        prompt_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_enc["input_ids"]
    pad_id = tokenizer.pad_token_id

    # Count non-pad prompt tokens
    prompt_ids = prompt_enc["input_ids"]
    prompt_len = sum(1 for t in prompt_ids if t != pad_id)

    # Labels: ignore prompt tokens, train only on completion
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    # Pad labels out to max_length with -100 (ignore padded tail)
    if len(labels) < max_length:
        labels = labels + [-100] * (max_length - len(labels))
    else:
        labels = labels[:max_length]

    assert len(labels) == len(input_ids) == max_length

    full_enc["labels"] = labels
    return full_enc


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        required=True)
    parser.add_argument("--num_train_epochs", type=float)
    parser.add_argument("--data_set", type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.num_train_epochs is not None:
        logger.info(f'updating num_train_epochs to {args.num_train_epochs}')
        config["training_params"]["num_train_epochs"] = args.num_train_epochs
    if args.data_set is not None:
        logger.info(f'updating data_set to {args.data_set}')
        config["data_set"] = args.data_set

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    data_formatted = load_and_format(config)
    data_tokenized = data_formatted.map(partial(tokenizer_func, tokenizer=tokenizer),
                                        remove_columns=data_formatted.column_names)
    if config['with_quantization']:
        bnb_config = BitsAndBytesConfig(**config['quantization_params'])
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            device_map="auto"
        )

    if config['with_lora']:
        lora_config = LoraConfig(**config['lora_params'])
        model_to_train = get_peft_model(model, lora_config)
    else:
        model_to_train = model

    training_args = TrainingArguments(**config['training_params'])

    trainer = Trainer(
        model=model_to_train,
        args=training_args,
        train_dataset=data_tokenized,
        tokenizer=tokenizer
    )

    trainer.train()

    output_dir = get_model_dir(config['task'])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir.as_posix())

if __name__ == '__main__':
    train()

