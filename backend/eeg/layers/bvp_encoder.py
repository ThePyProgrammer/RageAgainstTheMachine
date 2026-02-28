from torch import nn

class TrainableBVPEncoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout_rate: float = 0.1):
        self.block1 = self.build_bvp_encoder_block(
            in_features=in_features,
            proj_dim=256,
            dropout_rate=dropout_rate
        )
        self.block2 = self.build_bvp_encoder_block(
            in_features=256,
            proj_dim=512,
            dropout_rate=dropout_rate
        )
        self.block3 = self.build_bvp_encoder_block(
            in_features=256,
            proj_dim=512,
            dropout_rate=dropout_rate
        )
        
        self.final_proj = nn.Sequential(
            nn.Linear(512, out_features),
            nn.LayerNorm(out_features)
        )

    def build_bvp_encoder_block(self, 
                                in_features: int,
                                proj_dim: int,
                                dropout_rate: float = 0.1):
        return nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x_enc = self.block1(x)
        x_enc = self.block2(x_enc)
        x_enc = self.block3(x_enc)
        out = self.final_proj(x_enc)
        return out
