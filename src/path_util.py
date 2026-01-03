from pathlib import Path

cd = Path(__file__)

def get_data_path(task_name, data_name)->Path:
    return cd.parent.parent/'data'/task_name/(data_name+'.jsonl')

def get_model_dir(config)->Path:
    return cd.parent.parent / 'model' / config['task']/ f"{config['model_name']}_{config['train_set']}"

def get_output_dir(task_name)->Path:
    return cd.parent.parent / 'output' / task_name

def get_hl_cache_dir()->Path:
    path = cd.parent.parent/'hf_cache'
    path.mkdir(exist_ok=True, parents=True)
    return path

