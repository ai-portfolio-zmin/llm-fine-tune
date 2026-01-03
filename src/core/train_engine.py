import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import yaml
from functools import partial
from src.core.util import load_and_format, tokenizer_func
from src.path_util import get_model_dir, get_hl_cache_dir
from src.logger_util import get_logger

logger = get_logger('train')


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument("--num_train_epochs", type=float)
    parser.add_argument("--train_set", type=str)
    parser.add_argument("--learning_rate", type=float)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.num_train_epochs is not None:
        logger.info(f'updating num_train_epochs to {args.num_train_epochs}')
        config["training_params"]["num_train_epochs"] = args.num_train_epochs
    if args.train_set is not None:
        logger.info(f'updating train_set to {args.train_set}')
        config["train_set"] = args.train_set
    if args.learning_rate is not None:
        logger.info(f'updating learning_rate to {args.learning_rate}')
        config["training_params"]["learning_rate"] = args.learning_rate

    logger.info(f'config:\n{json.dumps(config, indent=2)}')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], cache_dir=get_hl_cache_dir().as_posix())
    data_formatted = load_and_format(config, 'train_set')
    logger.info(f'sample prompt: {data_formatted[0]["prompt"]}')
    logger.info(f'sample target: {data_formatted[0]["target"]}')
    data_tokenized = data_formatted.map(partial(tokenizer_func, tokenizer=tokenizer),
                                        remove_columns=data_formatted.column_names,
                                        load_from_cache_file=False)
    if config['with_quantization']:
        bnb_config = BitsAndBytesConfig(**config['quantization_params'])
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=get_hl_cache_dir().as_posix()
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            device_map="auto",
            cache_dir=get_hl_cache_dir().as_posix()
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

    output_dir = get_model_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'saving config down to {output_dir.as_posix()}')
    with open(output_dir / 'config.json', 'w') as f:
        f.write(json.dumps(config))
    logger.info(f'saving model down to {output_dir.as_posix()}')
    trainer.save_model(output_dir.as_posix())
    logger.info(f'List of files in {output_dir.as_posix()}:')
    for p in output_dir.iterdir():
        logger.info(p)


if __name__ == '__main__':
    train()
