import argparse
import json
import yaml
from src.path_util import get_model_dir, get_output_dir
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.core.util import load_and_format
from functools import partial
import importlib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from src.logger_util import get_logger

logger = get_logger('eval_engine')


def tokenizer_func(ex, tokenizer, model):
    return tokenizer(ex['prompt'], return_tensors="pt").to(model.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_dir = get_model_dir(config['task'])

    logger.info(f'loading {config["model_name"]}')
    base = AutoModelForCausalLM.from_pretrained(config['model_name'])
    logger.info(f'loading PEFT')
    model = PeftModel.from_pretrained(base, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    logger.info(f'formatting data')
    data_formatted = load_and_format(config)
    data_tokenized = data_formatted.map(partial(tokenizer_func, tokenizer=tokenizer, model=model),
                                        remove_columns=data_formatted.column_names)
    eval_func = importlib.import_module(f'src.tasks.{config["task"]}.eval')

    y_true = []
    y_pred = []
    i=0
    for data_formatted_i, data_tokenized_i in zip(data_formatted, data_tokenized):
        if i%50==0:
            logger.info(f'generating based on prompt {data_formatted_i["prompt"]}')
        output_i = model.generate(**data_tokenized_i,
                                  temperature=0.0,
                                  do_sample=False,
                                  **config['generate_params'], )
        full_tokens = output_i[0]
        prompt_len = data_tokenized_i['input_ids'].shape[1]
        gen_tokens = full_tokens[prompt_len:]
        output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        if i%50==0:
            logger.info(f'output text {output_text}')
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
        else:
            logger.error(f'Output not passing validation: {output_dict}'
                         f'Prompt:{data_formatted_i["prompt"]}')
            continue

    result = {}
    labels = config['labels']
    result['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    result['accuracy'] = accuracy_score(y_true, y_pred)
    result['f1'] = f1_score(y_true, y_pred, average="macro")
    result['recall'] = recall_score(y_true, y_pred, average="macro")

    logger.info(f'eval result: {result}')
    output_file = get_output_dir(config['task']) / 'result.json'
    with open(output_file, 'w') as f:
        f.write(json.dumps(result))


if __name__ == '__main__':
    main()
