import math
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange


def draw_convertible_width():
    def generate_B(a, shape):
        # 生成一个从0到1的等差数列
        x = torch.linspace(0, 1, shape[1])
        
        # 计算sigmoid函数
        B = torch.sigmoid((x - a / shape[1]) * 100)
        
        # 反转并归一化，使得在a之前的值接近1，在a之后的值接近0
        B = 1 - B
        
        return B.unsqueeze(0)  # 增加一个维度，使得B的shape为[1, 100]

    # 示例使用
    a = 50
    B = generate_B(a, shape=[1, 100]).tolist()[0]

    import matplotlib.pyplot as plt

    x = torch.linspace(0, 99, 100, dtype=int).tolist()

    plt.plot(x, B)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(True)
    plt.savefig('function_plot.png')


torch.manual_seed(5)

vocab_size = 100
d_model = 4
nhead = 2

batch_size = 2
seq_length = 50

num_epochs = 500
learning_rate = 0.1

x = torch.randint(0, vocab_size, (batch_size, seq_length)).to("cuda")
target = torch.ones(size=(batch_size, seq_length, d_model)).to("cuda")
# target = torch.rand(batch_size, seq_length, d_model)


class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)


        # Einsum does matrix multiplication for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just a way to do batch matrix multiplication
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # 2, 2, 5, 5
        if mask is not None:
            energy = energy * mask

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class SimpleModel(nn.Module):
    def __init__(self, d_model, nhead, vocab_size):
        super(SimpleModel, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.attn = Attention(self.d_model, self.nhead)
        self.converable_width = nn.Linear(self.d_model, 1)

    def position_embedding(self, x):
        batch_size, seq_length = x.shape
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)

        return pe.to("cuda")

    @staticmethod
    def min_max_normalize(x):
        return (x - torch.min(x, dim=-1, keepdim=True)[0]) / (torch.max(x, dim=-1, keepdim=True)[0] - torch.min(x, dim=-1, keepdim=True)[0])

    def converable_mask(self, x):
        bs, seq_length = x.shape[:-1]
        width = self.converable_width(x).squeeze(-1)  # bs, seq_length
        width = self.min_max_normalize(width).unsqueeze(-1)  # bs, seq_length, 1

        origin_mask = torch.cat([torch.linspace(1, 0, seq_length // 2), torch.linspace(0, 1, seq_length - seq_length // 2)]).to("cuda")
        origin_mask = origin_mask.unsqueeze(0).expand(bs, -1).unsqueeze(1)  # bs, 1, seq_length
        origin_mask = 1 - torch.sigmoid((origin_mask - width) * 100)  # bs, seq_length, seq_length

        indices = torch.arange(seq_length-1, -1, -1).unsqueeze(1).to("cuda") + torch.arange(seq_length).unsqueeze(0).to("cuda")
        indices = indices % seq_length
        indices = torch.roll(indices, shifts=math.ceil(seq_length/2), dims=0)

        move_mask = indices >= torch.roll(torch.arange(seq_length).unsqueeze(0), shifts=math.ceil(seq_length/2), dims=0).to("cuda")
        move_mask = torch.cat([move_mask[:math.ceil(seq_length/2)], ~move_mask[math.ceil(seq_length/2):]])
        
        result_mask = torch.zeros_like(origin_mask)
        result_mask = origin_mask[:, torch.arange(seq_length).unsqueeze(1), indices]
        result_mask = result_mask * move_mask

        return result_mask

    def forward(self, x, mask=None):
        x = self.embedding(x) + self.position_embedding(x)
        mask = self.converable_mask(x)
        x = self.attn(x, x, x, mask=mask)
        return x, mask


simple_model = SimpleModel(d_model, nhead, vocab_size).to("cuda")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(simple_model.parameters()), lr=learning_rate)

for epoch in trange(num_epochs):
    output, mask = simple_model(x)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if (epoch + 1) % 10 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if epoch+1 == 10:
        epoch_10 = mask[0, :, :].cpu().detach().numpy()
    if epoch+1 == 500:
        epoch_500 = mask[0, :, :].cpu().detach().numpy()

plt.figure(figsize=(10, 8))
sns.heatmap((np.abs(epoch_500-epoch_10) > 0.5).astype(int), annot=True, cmap='coolwarm')
plt.title('Heatmap of Scores')
plt.savefig(f'heatmap.png')

print("Training completed.")
