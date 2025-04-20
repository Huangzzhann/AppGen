import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class diff_model(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    cond_dim=config["cond_dim"],
                    total_dim=config["total_dim"],
                    appemb_dim=config["appemb_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

        self.skip_projection = nn.Conv1d(self.channels, self.channels, 1)
        self.output_projection = nn.Conv1d(self.channels, 1, 1)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, hist_info, ST_info, diffusion_step):
        B, inputdim, emb_dim = x.shape

        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, hist_info, ST_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)  # (B,channel,emb_dim)
        x = F.relu(x)
        x = self.output_projection(x)  # (B,1,emb_dim)
        x = x.reshape(B, emb_dim)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, cond_dim, total_dim, appemb_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.hist_cond_projection = nn.Sequential(
            CondUpsampler(total_dim, appemb_dim),
            Conv1d_with_init(1, 2 * channels, 1)
        )

        self.ST_cond_projection = Conv1d_with_init(cond_dim, 2 * channels, 1)

        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

    def forward(self, x, hist_info, ST_info, diffusion_emb):
        B, channel, emb_dim = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.mid_projection(y)  # (B,2*channel,L * emb_dim)

        # hist_info = self.hist_cond_projection(hist_info)
        # y = y + hist_info

        ST_info = self.ST_cond_projection(ST_info)
        y = y + ST_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,L * emb_dim)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class CrossAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.01):
        super().__init__()

        self.dropout = dropout

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=self.dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 对每一个batch进行归一化

    def forward(self, q, k, v, base_shape):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        B, C, emb_dim, L = base_shape

        residual = q

        q = self.layer_norm(q)

        # Pass through the pre-attention projection: batch_size x K*L x (n_head * dv)
        # Separate different heads: b x K*L x n x dv
        q = self.w_qs(q).view(B, C * emb_dim, n_head, d_k)
        k = self.w_ks(k).view(B, C * emb_dim, n_head, d_k)
        v = self.w_vs(v).view(B, C * emb_dim, n_head, d_v)

        # Transpose for attention dot product: b x n_head x K*L x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(B, C * emb_dim, -1)
        q = F.dropout(self.fc(q), self.dropout, training=self.training).reshape(base_shape)
        q += residual

        return q


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        attn = F.dropout(F.softmax(attn, dim=-1), self.dropout, training=self.training)
        output = torch.matmul(attn, v)

        return output
