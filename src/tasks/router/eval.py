


def validate(predict):
    type_check = isinstance(predict,dict)
    len_check = len(predict) ==1
    key_check = 'route' in predict
    value = predict.get('route','NA')
    value_check = value in ['bm25','hybrid','dense']
    return all((type_check,len_check,key_check, value_check))

def get_label(predict):
    return predict['route']