import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        # position embedding 的大小，通常根據 d_model 決定
        d_model = (d_model+1)//2*2  # 維度進位成偶數
        self.positional_encoding = torch.zeros(seq_len, d_model)
        
        # 計算位置索引和頻率指數
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(-(torch.log(torch.tensor(10000.0)) * torch.arange(0, d_model, 2) / d_model))
        
        # sin 與 cos 交替填充進位置嵌入
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)  # 增加 batch 維度
    
    def forward(self, x:torch.Tensor):
        return x + self.positional_encoding[:, :x.size(1), :x.size(2)].to(device=x.get_device(), dtype=torch.float32)

class ResidualFF(nn.Module):
    def __init__(self, n_channel, h_channel=512):
        super(ResidualFF, self).__init__()

        self.ff1 = nn.Linear(n_channel, h_channel, bias=False)
        self.actfn = nn.SiLU(inplace=True)
        self.ff2 = nn.Linear(h_channel, n_channel, bias=True)

    def forward(self, x):
        out = self.actfn(self.ff1(x))
        out = self.ff2(out)
        return out + x

class FeedForward(nn.Module):
    def __init__(self, in_channel, out_channel=512, h_channel=2048):
        super(FeedForward, self).__init__()

        self.ff1 = nn.Linear(in_channel, h_channel, bias=False)
        self.actfn = nn.SiLU(inplace=True)
        self.ff2 = nn.Linear(h_channel, out_channel, bias=True)

    def forward(self, x):
        out = self.actfn(self.ff1(x))
        out = self.ff2(out)
        return out

class ForcastingTransformer(nn.Module):
    def __init__(self, input_dim:int, store_dim=10, date_dim=11, model_dim=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super(ForcastingTransformer, self).__init__()

        self.position_encoding = nn.Sequential(
            ResidualFF(input_dim),
            PositionalEncoding(input_dim, seq_len=100),
        )

        self.date_position_encoding = nn.Sequential(
            PositionalEncoding(date_dim, seq_len=100),
            FeedForward(date_dim, model_dim, model_dim),
        )

        self.input_fc = FeedForward(input_dim, model_dim)
        self.encoder_Store = FeedForward(store_dim, model_dim, model_dim)

        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.output_fc = nn.Sequential(
            FeedForward(model_dim, input_dim),
            ResidualFF(input_dim, h_channel=512),
            nn.ReLU(inplace=True)
        )

    def forward(self, input, store, date, out_seqlen=7):
        # position embedding
        input = self.position_encoding(input)
        input = self.input_fc(input)

        # encode
        store = self.encoder_Store(store).unsqueeze_(1)
        date = self.date_position_encoding(date)
        enc_date = date[:,:-out_seqlen,...]
        dec_date = date[:,-out_seqlen:,...]
        
        # Transformer
        output = self.transformer(torch.cat([store, input+enc_date], dim=1), dec_date)
        
        # output layer
        output = self.output_fc(output)
        return output
