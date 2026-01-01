import importlib

def load_and_format(config):
    dataset = importlib.import_module(f'src.tasks.{config["task"]}.dataset')
    data = dataset.load_data(config['data_set'])
    data_formatted = data.map(dataset.format_example, remove_columns=data.column_names)
    return data_formatted
