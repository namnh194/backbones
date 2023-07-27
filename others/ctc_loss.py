import torch
import torch.nn as nn
import torch.nn.functional as F

class CTC_loss(nn.Module):
    def __init__(self):
        super(CTC_loss, self).__init__()
    def forward(self, x):
        return
class RNN_(nn.Module):
    def __init__(self):
        super(RNN_, self).__init__()
        self.input = nn.Embedding(num_embeddings=128, embedding_dim=128, padding_idx=0)
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        out = self.input(x)
        out, (h, c) = self.rnn(out)
        print(out[:, -1, :].view(out[:, -1, :].shape[0], -1) == h[-1,:,:].view(1,-1))
        print("out shape:", out.shape)
        print("hidden shape:", h.shape)
        print("cell shape:", c.shape)
        out = out[:, -1, :]
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
if __name__ == "__main__":
    inp = torch.randint(0, 128, (1,128)) # batch x seq_len x emb_dim
    net = RNN_()
    out = net(inp)
    print(out.shape)
    print(out)