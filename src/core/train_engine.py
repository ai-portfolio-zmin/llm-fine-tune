import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import yaml
from functools import partial
from src.core.util import load_and_format
from src.path_util import get_model_dir

def tokenizer_func(ex, tokenizer):
    full_text = ex['prompt'] + ex['target']
    prompt_text = ex['prompt']
    full_text_tokenized = tokenizer(full_text)
    prompt_text_tokenized = tokenizer(prompt_text)
    prompt_token_len = len(prompt_text_tokenized['input_ids'])
    labels = [-100] * prompt_token_len + full_text_tokenized['input_ids'][prompt_token_len:]
    full_text_tokenized['labels'] = labels
    return full_text_tokenized


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config['model'])
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

