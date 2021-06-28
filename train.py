import os
from transformers import BertTokenizer
from tqdm import tqdm,trange

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Optimizer,Adam,SGD,Adagrad


from utils import get_parser,show_ml_metric
from data import convert_to_inputdata, read_data, trans_inputdata
from models import Bert4TC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def warmup_linear(x, warmup = 0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def train():

    args = get_parser()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # prepare data
    train_text_list, train_label_list = read_data(args.train_file)
    eval_text_list, eval_label_list = read_data(args.eval_file)

    train_InputData_list = convert_to_inputdata(tokenizer,text_list_x = train_text_list, label_list = train_label_list, mode = "single")
    eval_InputData_list = convert_to_inputdata(tokenizer,text_list_x = eval_text_list, label_list = eval_label_list, mode = "single")

    print("len(train_InputData_list):",len(train_InputData_list))
    print("len(eval_InputData_list):", len(eval_InputData_list))

    model = Bert4TC.from_pretrained(args.bert_model, num_labels = args.num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = AdamW()

    # train
    all_input_ids, all_input_masks, all_segment_ids, all_labels = trans_inputdata(train_InputData_list,mode = "train")

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    all_input_masks = torch.tensor(all_input_masks, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).to(device)
    all_labels = torch.tensor(all_labels, dtype=torch.long).to(device)

    train_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.batch_size)

    model.train()
    global_step = 0
    num_train_steps = args.epochs * len(all_input_ids) / args.batch_size

    for _ in trange(int(args.epochs), desc = "Epoch"):

        for step, batch in enumerate(tqdm(train_dataloader, desc = "Iteration")):

            input_ids, input_mask, segment_ids, label = batch

            loss = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label)

            loss.backward()

            lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 2 == 0:
                print("global_step:", global_step, "loss:", loss.detach().cpu().numpy())

    # save model
    model_to_save = model.module if hasattr(model, 'module') else model

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # eval
    all_input_ids, all_input_masks, all_segment_ids, all_labels = trans_inputdata(eval_InputData_list, mode = "eval")

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    all_input_masks = torch.tensor(all_input_masks, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).to(device)

    eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=args.predict_batch_size)

    model.eval()

    target_list = []
    score_list = []

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            batch_logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            logits = F.softmax(batch_logits.detach().cpu(), dim=1)

            result = torch.max(logits, dim=1, keepdims=False)
            pred_target = result.indices.tolist()
            pred_score = result.values.tolist()

            target_list += pred_target
            score_list += pred_score

    show_ml_metric(all_labels, target_list, score_list)

if __name__ == "__main__":

    train()
