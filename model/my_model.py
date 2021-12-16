import torch
from torch import nn
from transformers import BertModel


class RobertaMultiLable(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1024, 148)

    def forward(self, guid, input_ids, attention_mask=None, token_type_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask
        )
        # (batch, seq_len, hidden_size)
        encoder_out = outputs.last_hidden_state
        # (batch, hidden_size)
        encoder_out = torch.max(encoder_out, dim=1)[0].squeeze(dim=1)
        encoder_out = self.dropout(encoder_out)
        # (batch, num_labels)
        out_logits = self.fc(encoder_out)
        return guid, out_logits
