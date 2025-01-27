import torch
from torch import nn
from transformers import BertTokenizer


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
                batch_size, self.hidden_size, device=input.device
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


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = GRU(
        vocab_size=len(tokenizer),
        embedding_dim=256,
        hidden_dim=128,
        output_dim=1,
        n_layers=2,
    )
    model.load_state_dict(torch.load("best_GRU_with_NAdam.pth", weights_only=True, map_location='cpu')['model_state_dict'])
    model.eval()
    with torch.no_grad():
        x = input()
        x = tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        x = x["input_ids"][0]
        x = x.unsqueeze(0)
        x = x.to('cpu')
        pred = model(x)
        pred = torch.round(pred).item()
        print("positive" if pred == 1.0 else "negative")
