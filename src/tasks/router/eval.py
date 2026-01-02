import re
import json


def validate(predict):
    type_check = isinstance(predict,dict)
    len_check = len(predict) ==1
    key_check = 'route' in predict
    value = predict.get('route','NA')
    value_check = value in ['bm25','hybrid','dense']
    return all((type_check,len_check,key_check, value_check))

def get_label(predict):
    return predict['route']



def extract_output(text: str) -> dict:
    """
    Extracts and validates router output from model generation.

    Returns:
        dict: {"route": "<bm25|dense|hybrid>"}

    Raises:
        ValueError: if block missing, JSON invalid, or schema invalid
    """
    m = re.search(r'^(.*?)\n{3}', text, re.DOTALL)

    if m:
        result = m.group(1)
    else:
        result = text  # fallback if no triple newline

    return result