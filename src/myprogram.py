#!/usr/bin/env python
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(BASE_DIR, "../work")
SEQ_LEN  = 100
LEARNING_RATE = 3e-4

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, ff_dim=512, max_seq=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = nn.Embedding(max_seq, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # casual mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        # drop out to prevent overfitting
        out = self.dropout(self.embed(x) + self.pos_enc(positions))
        # run transformer
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.fc(out)

class CharDataset(Dataset):
    def __init__(self, texts, char2idx, seq_len, stride = 10):
        self.seq_len = seq_len
        self.char2idx = char2idx
        # Flatten all text into one big stream
        full = "\n".join(texts)
        self.data = [char2idx.get(c, 1) for c in full]
        self.indices = list(range(0, len(self.data) - seq_len, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = torch.tensor(self.data[start:start+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start+1:start+self.seq_len+1], dtype=torch.long)
        return x, y


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.model    = None
        self.char2idx = None
        self.idx2char = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load_training_data(cls):
        with open(os.path.join(WORK_DIR, "all_text.json"), "r", encoding="utf-8") as f:
            all_text = json.load(f)

        return all_text

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname, encoding="utf-8") as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding="utf-8") as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, work_dir):
        # create data loaders
        all_text = self.load_training_data()

        chars = sorted(set("".join(all_text)))
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        for c in chars:
            self.char2idx[c] = len(self.char2idx)
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        vocab_size = len(self.char2idx)
        print(f"Vocab size: {vocab_size}")

        with open(os.path.join(work_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx, "idx2char": self.idx2char}, f, ensure_ascii=False)

        dataset = CharDataset(all_text, self.char2idx, SEQ_LEN)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)

        model = Transformer(vocab_size)

        opt = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(20):
            model.train()
            total_loss = 0
            for i, (x, y) in enumerate(loader):
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                # avoid OOM
                del logits
                torch.cuda.empty_cache()
                total_loss += loss.item()

                if i % 200 == 0:
                    print(f"Epoch {epoch + 1} Step {i} Loss {loss.item():.4f}")

            sched.step()
            avg = total_loss / len(loader)
            print(f"Epoch {epoch + 1} avg loss: {avg:.4f}")

        self.save(work_dir)

    def run_pred(self, data):
        preds = []
        for inp in data:
            context = inp[-SEQ_LEN:]
            # get numbers or unknown number
            ids = [self.char2idx.get(c, 1) for c in context]
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x)
            last_logits = logits[0, -1]
            # removing pad and unknown tokens
            last_logits[0] = float('-inf')
            last_logits[1] = float('-inf')
            top3_ids = torch.topk(last_logits, 3).indices.tolist()
            top3_chars = [self.idx2char.get(i, "?") for i in top3_ids]
            preds.append("".join(top3_chars))

        return preds

    def save(self, work_dir):
        torch.save({
            "model_state": self.model.state_dict(),
            "vocab_size": len(self.char2idx),
            "embed_dim": self.model.embed.embedding_dim,
            "num_heads": self.model.transformer.layers[0].self_attn.num_heads,
            "num_layers": len(self.model.transformer.layers),
            "ff_dim": self.model.transformer.layers[0].linear1.out_features,
        }, os.path.join(work_dir, "model.pt"))

    @classmethod
    def load(cls, work_dir):
        m = cls()

        with open(os.path.join(work_dir, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)
        m.char2idx = vocab["char2idx"]
        m.idx2char = {int(k): v for k, v in vocab["idx2char"].items()}

        model_saved = torch.load(os.path.join(work_dir, "model.pt"), map_location=m.device)
        m.model = Transformer(
            vocab_size=model_saved["vocab_size"],
            embed_dim=model_saved["embed_dim"],
            num_heads=model_saved["num_heads"],
            num_layers=model_saved["num_layers"],
            ff_dim=model_saved["ff_dim"],
        )
        m.model.load_state_dict(model_saved["model_state"])
        m.model.eval()
        return m


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        model = MyModel()
        model.run_train(args.work_dir)

    elif args.mode == 'test':
        model = MyModel.load(args.work_dir)
        test_data = MyModel.load_test_data(args.test_data)
        preds = model.run_pred(test_data)
        model.write_pred(preds, args.test_output)