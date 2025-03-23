import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 8, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)
    

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(DownSample, self).__init__()
        self.downsample = nn.Sequential(nn.MaxPool2d(2, 2),
                                        DoubleConv(in_channels, out_channels),
                                        DoubleConv(out_channels , out_channels))   
        self.linear = nn.Sequential(nn.SiLU(inplace=True),
                                    nn.Linear(time_embedding_dim, out_channels))
               
    def forward(self, x, t):
        x = self.downsample(x)
        t = self.linear(t)
        t = t[:, :, None, None]
        t = t.expand(-1, -1, x.shape[2], x.shape[3])
        return x + t
    

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.doubleconv = nn.Sequential(DoubleConv(in_channels, out_channels),
                                        DoubleConv(out_channels , out_channels))   
        self.linear = nn.Sequential(nn.SiLU(inplace=True),
                                    nn.Linear(time_embedding_dim, out_channels))
               
    def forward(self, x, y, t):
        x = self.upsample(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([y, x], dim=1)
        x = self.doubleconv(x)
        t = self.linear(t)
        t = t[:, :, None, None]
        t = t.expand(-1, -1, x.shape[2], x.shape[3])
        return x + t
    

class AttentionHead(nn.Module):
    def __init__(self, in_channel, embedding_dim):
        super(AttentionHead, self).__init__()
        self.query = nn.Linear(in_channel, embedding_dim)
        self.key = nn.Linear(in_channel, embedding_dim)
        self.value = nn.Linear(in_channel, in_channel)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5))
        return torch.matmul(attention, v)
    

class SelfAttention(nn.Module):
    def __init__(self, in_channel, embedding_dim, number_of_heads):
        super(SelfAttention, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(in_channel, embedding_dim) for _ in range(number_of_heads)])
        self.linear = nn.Linear(in_channel * number_of_heads, in_channel)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x = F.layer_norm(x,[ x.shape[-1]])
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x
    
class TimeEmbbeding(nn.Module):
    def __init__(self, max_steps, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_steps, embedding_dim)

    def forward(self, t):
        return self.embedding(t)  # Shape: (batch_size, embedding_dim)

if __name__ == "__main__":
    x = torch.randint(0, 64, (1,1)).cuda()
    model = TimeEmbbeding(64, 32).cuda()
    print(model(x).shape)