from src.path_util import get_data_path
from src.tasks.router.prompt import Template
from datasets import load_dataset
import json


def format_example(example) -> dict:
    query = example['query']
    features = example['context']
    route = example['route']

    prompt = Template.format(query=query, features=json.dumps(features))
    target = json.dumps({'route': route})
    return {'prompt': prompt, 'target': target,'input':query}


def load_data(mode='train'):
    data = load_dataset('json', data_files=get_data_path('router', mode).as_posix())['train']
    return data


if __name__ == '__main__':
    data_ex = load_data()
    print(data_ex.map(format_example,remove_columns=data_ex.column_names))