import argparse
import json
import yaml
from src.path_util import get_model_dir, get_output_dir, get_hl_cache_dir
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.core.util import load_and_format
from functools import partial
import importlib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from src.logger_util import get_logger
import torch

logger = get_logger('eval_engine')


def tokenizer_func(ex, tokenizer, max_length=512):
    return tokenizer(
        ex["prompt"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        required=True)
    parser.add_argument("--eval_set", type=str)
    parser.add_argument("--train_set", type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.eval_set is not None:
        logger.info(f'updating eval_set to {args.eval_set}')
        config["eval_set"] = args.eval_set
    if args.train_set is not None:
        logger.info(f'updating train_set to {args.train_set}')
        config["train_set"] = args.train_set
    max_len = config.get("max_length", 512)
    model_dir = get_model_dir(config)

    logger.info(f'loading {config["model_name"]} on GPU')
    if config['with_quantization']:
        bnb_config = BitsAndBytesConfig(**config['quantization_params'])
        base = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=get_hl_cache_dir().as_posix()
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            device_map="auto",
            torch_dtype="bfloat16",
            cache_dir=get_hl_cache_dir().as_posix()
        )
    logger.info(f'loading adaptor from {model_dir}')
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"],
                                              cache_dir=get_hl_cache_dir().as_posix()
                                              )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f'formatting data')
    data_formatted = load_and_format(config,'eval_set')
    data_tokenized = data_formatted.map(
        partial(tokenizer_func, tokenizer=tokenizer, max_length=max_len),
        remove_columns=data_formatted.column_names,
        load_from_cache_file=False
    )
    eval_func = importlib.import_module(f'src.tasks.{config["task"]}.eval')
    y_true = []
    y_pred = []
    user_input = []
    gen_params = config.get("generate_params", {})
    gen_params.setdefault("max_new_tokens", 32)
    gen_params.setdefault("pad_token_id", tokenizer.eos_token_id)

    i = 0
    for data_formatted_i, data_tokenized_i in zip(data_formatted, data_tokenized):
        # log occasionally
        if i % 50 == 0:
            logger.info(f'generating for prompt (truncated): {data_formatted_i["prompt"][:200]}')
        i += 1

        # move inputs to GPU
        inputs = {k: torch.tensor(v).to(model.device) for k, v in data_tokenized_i.items()}

        # generation
        with torch.no_grad():
            output_i = model.generate(
                **inputs,
                temperature=0.0,
                do_sample=False,
                **gen_params,
            )

        full_tokens = output_i[0]
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = full_tokens[prompt_len:]

        output_text_long = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        output_text = eval_func.extract_output(output_text_long)
        if i % 50 == 0:
            logger.info(f'output text (truncated): {output_text[:200]}')
        try:
            output_dict = json.loads(output_text)
        except:
            logger.error(f'error loading using json: {output_text}')
            logger.error(f'Prompt:{data_formatted_i["prompt"]}')
            continue
        validate_i = eval_func.validate(output_dict)
        if validate_i:
            if isinstance(data_formatted_i['target'], str):
                target_dict = json.loads(data_formatted_i['target'])
            else:
                target_dict = data_formatted_i['target']
            y_pred.append(eval_func.get_label(output_dict))
            y_true.append(eval_func.get_label(target_dict))
            user_input.append(data_formatted_i['input'])
        else:
            logger.error(f'Output not passing validation: {output_dict}'
                         f'Prompt:{data_formatted_i["prompt"]}')
            continue

    gen_result = [
        {"input": k, "target": target, "predict": predict}
        for k, target, predict in zip(user_input, y_true, y_pred)
    ]
    gen_output_file = get_output_dir(config['task']) / f'{config["train_set"]}_{config["eval_set"]}_gen_result.json'
    gen_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(gen_output_file, 'w') as f:
        f.write(json.dumps(gen_result))

    result = {}
    labels = config['labels']
    result['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    result['accuracy'] = accuracy_score(y_true, y_pred)
    result['f1'] = f1_score(y_true, y_pred, average="macro")
    result['recall'] = recall_score(y_true, y_pred, average="macro")

    logger.info(f'eval result: {result}')
    output_file = get_output_dir(config['task']) / f'{config["train_set"]}_{config["eval_set"]}_result.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(json.dumps(result))


if __name__ == '__main__':
    main()
