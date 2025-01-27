# %%
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
df = pd.read_csv("data/IMDB Dataset.csv")

df

# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = self.df.iloc[idx]["review"]
        label = self.df.iloc[idx]["sentiment"]
        tokens = self.tokenizer(
            review,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        label = torch.tensor(1.0 if label == "positive" else 0.0)
        return tokens["input_ids"][0], label


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


train_dataset = ImdbDataset(train_df, tokenizer)
val_dataset = ImdbDataset(val_df, tokenizer)

batch_size = 1024

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
)


# %%
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hx=None):
        batch_size, _ = input.size()
        if hx is None:
            hx = (
                torch.zeros(batch_size, self.hidden_size, device=input.get_device()),
                torch.zeros(batch_size, self.hidden_size, device=input.get_device()),
            )
        hidden, cell = hx

        combined = torch.cat((input, hidden), 1)
        gates = self.gate(combined)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        f_gate = torch.sigmoid(f_gate)
        i_gate = torch.sigmoid(i_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        cell = torch.add(torch.mul(f_gate, cell), torch.mul(i_gate, g_gate))
        hidden = torch.mul(o_gate, torch.tanh(cell))
        hidden = self.dropout(hidden)
        return hidden, cell


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.ModuleList(
            [
                (
                    LSTMCell(
                        embedding_dim, hidden_dim, 0 if layer == n_layers - 1 else 0.2
                    )
                    if layer == 0
                    else LSTMCell(
                        hidden_dim, hidden_dim, 0 if layer == n_layers - 1 else 0.2
                    )
                )
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.embedding(input)
        _, seq_size, _ = x.size()
        hidden = [None for _ in range(len(self.rnn))]
        for t in range(seq_size):
            input_t = x[:, t, :]
            for layer_idx in range(len(self.rnn)):
                hidden[layer_idx] = self.rnn[layer_idx](input_t, hidden[layer_idx])
                input_t = hidden[layer_idx][0]
        x = input_t
        x = self.dropout(x)
        x = self.fc(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        return x


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, 2 * hidden_size, bias=True)
        self.gate_i = nn.Linear(input_size, hidden_size, bias=True)
        self.gate_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden=None):
        batch_size, _ = input.size()
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.hidden_size, device=input.get_device()
            )
        combined = torch.cat((input, hidden), 1)

        gates = self.gate(combined)
        r_gate, z_gate = gates.chunk(2, dim=1)
        r_gate = torch.sigmoid(r_gate)
        z_gate = torch.sigmoid(z_gate)
        i_gate = self.gate_i(input)
        h_gate = self.gate_h(hidden)
        n_gate = torch.add(i_gate, torch.mul(r_gate, h_gate))
        n_gate = torch.tanh(n_gate)
        hidden = torch.add(
            torch.mul(torch.sub(1, z_gate), n_gate), torch.mul(z_gate, hidden)
        )
        hidden = self.dropout(hidden)
        return hidden


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.ModuleList(
            [
                (
                    GRUCell(
                        embedding_dim, hidden_dim, 0 if layer == n_layers - 1 else 0.2
                    )
                    if layer == 0
                    else GRUCell(
                        hidden_dim, hidden_dim, 0 if layer == n_layers - 1 else 0.2
                    )
                )
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.embedding(input)
        _, seq_size, _ = x.size()
        hidden = [None for _ in range(len(self.rnn))]
        for t in range(seq_size):
            input_t = x[:, t, :]
            for layer_idx in range(len(self.rnn)):
                hidden[layer_idx] = self.rnn[layer_idx](input_t, hidden[layer_idx])
                input_t = hidden[layer_idx]
        x = input_t
        x = self.dropout(x)
        x = self.fc(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        return x


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden=None):
        batch_size, _ = input.size()
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.hidden_size, device=input.get_device()
            )
        combined = torch.cat((input, hidden), 1)
        h_gate = self.gate(combined)
        h_gate = torch.tanh(h_gate)
        h_gate = self.dropout(h_gate)
        return h_gate


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.ModuleList(
            [
                (
                    RNNCell(
                        embedding_dim, hidden_dim, 0 if layer == n_layers - 1 else 0.2
                    )
                    if layer == 0
                    else RNNCell(
                        hidden_dim, hidden_dim, 0 if layer == n_layers - 1 else 0.2
                    )
                )
                for layer in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.embedding(input)
        _, seq_size, _ = x.size()
        hidden = [None for _ in range(len(self.rnn))]
        for t in range(seq_size):
            input_t = x[:, t, :]
            for layer_idx in range(len(self.rnn)):
                hidden[layer_idx] = self.rnn[layer_idx](input_t, hidden[layer_idx])
                input_t = hidden[layer_idx]
        x = input_t
        x = self.dropout(x)
        x = self.fc(x)
        x = x.squeeze()
        x = torch.sigmoid(x)
        return x


# %%
class Model:
    def __init__(self, model, optimizer, criterion, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_valid = float(0)
        if isinstance(self.model, LSTM):
            self.name = "LSTM"
        elif isinstance(self.model, GRU):
            self.name = "GRU"
        elif isinstance(self.model, RNN):
            self.name = "RNN"
        self.lr = []

    def optim_to_str(self):
        if isinstance(self.optimizer, torch.optim.SGD):
            if self.optimizer.param_groups[0]["nesterov"]:
                return "NAG"
            else:
                return "SGD"
        # if isinstance(self.optimizer, torch.optim.Adadelta):
        #     return "Adadelta"
        if isinstance(self.optimizer, torch.optim.NAdam):
            return "NAdam"
        if isinstance(self.optimizer, torch.optim.RAdam):
            return "RAdam"
        if isinstance(self.optimizer, torch.optim.Adamax):
            return "Adamax"
        if isinstance(self.optimizer, torch.optim.AdamW):
            return "AdamW"

    def __str__(self):

        return self.name + "_with_" + self.optim_to_str()

    def train_one_epoch(self, trainloader):
        self.model.train()
        print(f"Training {self}")
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        clip = 5
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            train_running_loss += loss.item()
            preds = torch.round(outputs.data)
            train_running_correct += (preds == labels).sum().item()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
        self.train_losses.append(train_running_loss / counter)
        self.train_accs.append(
            100.0 * (train_running_correct / len(trainloader.dataset))
        )

    def validate(self, testloader):
        self.model.eval()
        print(f"Validation {self}")
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                counter += 1
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                valid_running_loss += loss.item()
                preds = torch.round(outputs.data)
                valid_running_correct += (preds == labels).sum().item()
        self.val_losses.append(valid_running_loss / counter)
        self.val_accs.append(100.0 * (valid_running_correct / len(testloader.dataset)))

    def step(self, epoch, train_loader, val_loader):
        self.train_one_epoch(train_loader)
        self.validate(val_loader)
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(self.val_accs[-1])
        else:
            self.lr_scheduler.step()
        print(
            f"Training loss: {self.train_losses[-1]:.3f}, training acc: {self.train_accs[-1]:.3f}"
        )
        print(
            f"Validation loss: {self.val_losses[-1]:.3f}, validation acc: {self.val_accs[-1]:.3f}"
        )
        print(f"Lr: {self.lr_scheduler.get_last_lr()[0]}")
        self.lr.append(self.lr_scheduler.get_last_lr()[0])
        self.save_best(epoch)
        print("-" * 50)

    def save_best(self, epoch):
        if self.val_accs[-1] > self.best_valid:
            self.best_valid = self.val_accs[-1]
            print(f"\nBEST VALIDATION: {self.best_valid}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                },
                f"best_{self}.pth",
            )

    def save_model(self, epochs):
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.criterion,
            },
            f"{self}.pth",
        )

    def save_train_data(self, epochs):
        data = {
            "name": str(self),
            "epochs": epochs,
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "best_valid": self.best_valid,
            "lr": self.lr,
        }
        with open(f"{self}_dat.pkl", "wb") as f:
            pickle.dump(data, f)


# %%
vocab_size = len(tokenizer)
embedding_dim = 256
hidden_dim = 128
output_dim = 1
n_layers = 2
models = []
models_classes = [LSTM, GRU, RNN]
opt_classes = [
    torch.optim.SGD,
    torch.optim.Adamax,
    torch.optim.NAdam,
    torch.optim.RAdam,
]
weight_decay = 1e-4
for model_cl in models_classes:
    for opt_cl in opt_classes:

        model = model_cl(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
        )
        match opt_cl:
            case torch.optim.SGD:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    weight_decay=weight_decay,
                    momentum=0.9,
                    nesterov=True,
                )
            case torch.optim.Adamax:
                optimizer = torch.optim.Adamax(
                    model.parameters(), weight_decay=weight_decay
                )
            case torch.optim.NAdam:
                optimizer = torch.optim.NAdam(
                    model.parameters(),
                    decoupled_weight_decay=True,
                    weight_decay=weight_decay,
                )
            case torch.optim.RAdam:
                optimizer = torch.optim.RAdam(
                    model.parameters(),
                    decoupled_weight_decay=True,
                    weight_decay=weight_decay,
                )
        criterion = torch.nn.BCELoss()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, eta_min=1e-5
        )
        models.append(Model(model, optimizer, criterion, lr_scheduler))


# %%
def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))


# %%
[print_model_size(model.model) for model in models[::4]]

# %%
epochs = 20
for model in models[4:]:
    print(f"[INFO]: {model}")
    model.model.to(device)
    for epoch in range(0, epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        model.step(epoch, train_dataloader, val_dataloader)
    model.save_model(epoch)
    model.save_train_data(epoch)
    del model.model
    torch.cuda.empty_cache()
