import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_twitter_train, load_twitter_val
from preprocessing import clean_text
from sentiment_model import RoBERTaGRU, train_epoch

def main():
    # Load & preprocess raw text from CSVs
    train_df = load_twitter_train('data/twitter_training.csv')
    val_df   = load_twitter_val(  'data/twitter_validation.csv')

    train_df['clean'] = train_df['text'].map(clean_text)
    val_df[  'clean'] = val_df[  'text'].map(clean_text)

    # Remap sentiment strings to integer classes 0–3
    label_map = {
        'Negative':   0,
        'Neutral':    1,
        'Positive':   2,
        'Irrelevant': 3
    }
    train_labels = train_df['sentiment'].map(label_map).values
    val_labels   = val_df[  'sentiment'].map(label_map).values

    train_texts = train_df['clean'].tolist()
    val_texts   = val_df[  'clean'].tolist()

    # Tokenization for RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    def encode_texts(text_list, max_len=128):
        return tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

    train_enc = encode_texts(train_texts)
    val_enc   = encode_texts(val_texts)

    # Build TensorDatasets & DataLoaders
    train_ds = TensorDataset(
        train_enc['input_ids'],
        train_enc['attention_mask'],
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_ds   = TensorDataset(
        val_enc['input_ids'],
        val_enc['attention_mask'],
        torch.tensor(val_labels,   dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    # Initialize model, optimizer, criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RoBERTaGRU(num_labels=4).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Warm-up batch timing 
    # Needed when testing
    start = time.time()
    for i, batch in enumerate(train_loader):
        input_ids, attn_mask, labels = [t.to(device) for t in batch]
        logits = model(input_ids, attn_mask)
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        if i == 49: break
    avg_batch = (time.time() - start) / 50
    total_batches = len(train_loader) * 3  # 3 epochs
    est_seconds = avg_batch * total_batches
    print(f"Avg batch time: {avg_batch:.3f}s → Estimated total training: {est_seconds/3600:.1f}h")

    # Training loop
    epochs = 3
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{epochs} done")
        
    print("Training complete.")

    # Evaluate and graph data
    def evaluate(model, loader, device):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for input_ids, attn_mask, labels in loader:
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
                logits = model(input_ids, attn_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # Classification report
        target_names = ['Negative', 'Neutral', 'Positive', 'Irrelevant']
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:\n", cm)

        # Plot confusion matrix
        fig, plot = plt.subplots()
        im = plot.imshow(cm)
        plot.set_xticks(np.arange(len(target_names)))
        plot.set_yticks(np.arange(len(target_names)))
        plot.set_xticklabels(target_names, rotation=45, ha='right')
        plot.set_yticklabels(target_names)
        plot.set_xlabel('Predicted')
        plot.set_ylabel('True')
        plot.set_title('Confusion Matrix')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plot.text(j, i, cm[i, j], ha='center', va='center')
        plt.tight_layout()
        plt.show()

    torch.save(model.state_dict(), 'roberta_gru_sentiment.pt')
    print("Model saved to roberta_gru_sentiment.pt")

    evaluate(model, val_loader, device)

if __name__ == '__main__':
    main()
