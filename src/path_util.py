from pathlib import Path

cd = Path(__file__)

def get_data_path(task_name, data_name):
    return cd.parent.parent/'data'/task_name/(data_name+'.jsonl')

def get_model_dir(task_name):
    return cd.parent.parent / 'model' / task_name

def get_output_dir(task_name):
    return cd.parent.parent / 'output' / task_name


