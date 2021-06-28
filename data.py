import torch
from utils import get_parser

class InputData(object):

    def __init__(self,
                 unique_id,
                 text,
                 tokens,
                 input_ids,
                 input_masks,
                 segment_ids = None,
                 label = None
                 ):
        self.unique_id = unique_id
        self.text = text
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.label = label
        self.segment_ids = segment_ids

def convert_to_inputdata(tokenizer, text_list_x, text_list_y = None, label_list = None, mode = "single"):

    '''
    :param text_list_x:
    :param text_list_y:
    :param mode: single or pair, single没问题, pair可能有bug
    :return: inputdata_list
    '''

    args = get_parser()
    max_seq_length = args.max_seq_length

    if mode == "single":
        assert text_list_y == None, "text_list_y must be None"
    elif mode == "pair":
        assert text_list_y != None, "text_list_y must be str_list"
        assert len(text_list_x) == len(text_list_y), "len(text_list_x) == len(text_list_y)"

    InputData_list = []

    for unique_id in range(len(text_list_x)):

        one_text_x = text_list_x[unique_id]
        if len(one_text_x) > max_seq_length - 2:
            one_text_x = one_text_x[:max_seq_length - 2]

        if label_list is not None:
            one_label = label_list[unique_id]
        else:
            one_label = None
        text = one_text_x

        if mode == "pair":
            one_text_y = text_list_y[unique_id]
            # ###分隔
            text = text + "###" + one_text_y

        tokens = ["[CLS]"]
        segment_ids = []

        # 这里是不需要判断位置的直接放入即可
        tokens.extend(tokenizer.tokenize(one_text_x))

        tokens.append("[SEP]")
        segment_ids += [0] * len(tokens)
        l_seg = len(tokens)


        if mode == "pair":
            tokens.extend(tokenizer.tokenize(one_text_y))
            tokens.append("[SEP]")

            segment_ids += [1] * (len(tokens) - l_seg)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_masks.append(0)
            segment_ids.append(0)

        one_inputdata = InputData(unique_id = unique_id,
                                     text = text,
                                     tokens = tokens,
                                     input_ids = input_ids,
                                     input_masks = input_masks,
                                     segment_ids = segment_ids,
                                     label = one_label)

        InputData_list.append(one_inputdata)

    return InputData_list

def trans_inputdata(InputData_list, mode = "train"):

    '''
    :param InputData_list:
    :param mode: train or eval or test, test -> 数据没有label
    :return:
    '''

    all_input_ids, all_input_masks, all_segment_ids = [], [], []

    if mode == "train" or mode == "eval":
        all_labels = []

    for one_inputdata in InputData_list:
        all_input_ids.append(one_inputdata.input_ids)
        all_input_masks.append(one_inputdata.input_masks)
        all_segment_ids.append(one_inputdata.segment_ids)
        if mode == "train" or mode == "eval":
            all_labels.append(one_inputdata.label)

    result = [all_input_ids,all_input_masks,all_segment_ids]

    if mode == "train" or mode == "eval":

        result.append(all_labels)

    return result


def read_data(filename):

    text_list = []
    label_list = []

    with open(filename,'r',encoding = 'utf-8') as f:
        for line in f.readlines():
            text, label = line.strip().split()
            text_list.append(text)
            label_list.append(int(label))

    assert len(text_list) == len(label_list)

    return text_list, label_list












