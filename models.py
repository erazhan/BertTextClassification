from transformers import BertPreTrainedModel,BertModel

from torch import nn
from torch.nn import CrossEntropyLoss

class Bert4TC(BertPreTrainedModel):

    def __init__(self,config):

        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size,self.num_labels)
        self.init_weights()

    def forward(self,
                input_ids = None,
                attention_mask = None,
                token_type_ids = None,
                position_ids = None,
                head_mask = None,
                inputs_embeds = None,
                labels = None,
                output_attentions = None,
                output_hidden_states = None,
                return_dict=None,
                ):

        outputs = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds,# 有版本后面是不执行的
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states,
                           return_dict=return_dict,)

        # outputs[0].shape: [bsz, maxlen, hidden_size]

        # outputs[1].shape: [bsz, hidden_size], 取的是outputs[0]的第一个向量, 然后经过一个dense层(同维), 再经过一个Tanh层激活函数层
        # pooler -> 平均,最大,cls. 平均的效果最好, 默认平均
        # 具体查看transformers中的models.bert.modeling_bert.BertPooler

        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        if labels is not None:

            loss_fact = CrossEntropyLoss()

            preds = logits.view(-1,self.num_labels)
            targs = labels.view(-1)

            loss = loss_fact(preds,targs)

            return loss

        return logits
