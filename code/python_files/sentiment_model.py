import torch, torch.nn as nn
from transformers import RobertaModel

class RoBERTaGRU(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.gru = nn.GRU(768, 256, batch_first=True)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask).last_hidden_state
        _, h = self.gru(outputs)
        return self.classifier(h.squeeze(0))

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for b in dataloader:
        input_ids, attn_mask, labels = [t.to(device) for t in b]
        logits = model(input_ids, attn_mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
