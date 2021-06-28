import argparse

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

# ----------------------------------------------------------------------------------------------------------------------

def read_txt_file(txt_filename):

    '''
    :param txt_filename: txt文件,每一行是一个实体
    :return: 实体列表
    注意这里不去重
    '''

    entity_list = []
    with open(txt_filename,encoding = 'utf-8',mode = 'r') as f:
        for line in f.readlines():
            entity_list.append(line.strip())

    return entity_list

def save_txt_file(filename,entity_list):

    with open(filename,encoding='utf-8',mode ='w') as f:
        for one_entity in entity_list:
            f.write(one_entity + '\n')

def show_ml_metric(test_labels, predict_labels, predict_prob):
    accuracy = accuracy_score(test_labels, predict_labels)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1_measure = f1_score(test_labels, predict_labels)
    confusionMatrix = confusion_matrix(test_labels, predict_labels)
    fpr, tpr, threshold = roc_curve(test_labels, predict_prob, pos_label=1)
    Auc = auc(fpr, tpr)
    print("------------------------- ")
    print("confusion matrix:")
    print("------------------------- ")
    print("| TP: %5d | FP: %5d |" % (confusionMatrix[1, 1], confusionMatrix[0, 1]))
    print("----------------------- ")
    print("| FN: %5d | TN: %5d |" % (confusionMatrix[1, 0], confusionMatrix[0, 0]))
    print(" ------------------------- ")
    print("Accuracy:       %.2f%%" % (accuracy * 100))
    print("Recall:         %.2f%%" % (recall * 100))
    print("Precision:      %.2f%%" % (precision * 100))
    print("F1-measure:     %.2f%%" % (f1_measure * 100))
    print("AUC:            %.2f%%" % (Auc * 100))
    print("------------------------- ")
    return (Auc)

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model",default = "/Users/zjy/Desktop/code/pretrained_model/chinese-roberta-wwm-ext",help = "预训练模型地址")

    parser.add_argument("--train_file", default = './data/train.txt',type = str, help = '训练数据')
    parser.add_argument("--eval_file", default = './data/eval.txt',type = str, help = '测试数据')

    parser.add_argument('--output_dir', default='./model', type=str, help='输出模型文件夹')
    parser.add_argument('--predict_dir', default='./model', type=str, help='预测模型文件夹')

    parser.add_argument('--max_seq_length',default = 256, type = int, help = '文本分类模型最大长度')

    parser.add_argument('--batch_size',default = 4, type = int, help = '训练batch大小')
    parser.add_argument('--predict_batch_size',default = 4, type = int, help = '预测时batch大小')
    parser.add_argument('--num_labels',default = 2,type = int, help = '0-负例，1-正例')

    parser.add_argument('--learning_rate',default = 2e-5, type = float, help = '学习率大小')
    parser.add_argument('--epochs',default = 3, type = int,help = '训练次数')
    parser.add_argument("--warmup_proportion", default = 0.1, type = float, help = "warmup比例")

    return parser.parse_args()

