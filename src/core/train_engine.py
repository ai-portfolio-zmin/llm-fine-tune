import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import yaml
from functools import partial
from src.core.util import load_and_format
from src.path_util import get_model_dir
from src.logger_util import get_logger

logger = get_logger('train')

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
    logger.info(f'sample prompt: {data_formatted[0]["prompt"]}')
    logger.info(f'sample target: {data_formatted[0]["target"]}')
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
        tokenizer=tokenizer,
    )
    logger.info('training starts')
    trainer.train()

    output_dir = get_model_dir(config['task'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'saving model down to {output_dir.as_posix()}')
    trainer.save_model(output_dir.as_posix())

if __name__ == '__main__':
    train()