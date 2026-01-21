import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torch

class VideoBleedingDetectorLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256, dropout_rate=0.3):
        super().__init__()

        # Keep full R2Plus1D backbone
        full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(full_model.children())[:-2])

        # Process spatial-temporal features
        self.temporal_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep temporal dim

        # LSTM on temporal features
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Heads
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.volume_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: [B, C, T, H, W]
        # Extract spatio-temporal features
        features = self.backbone(x)  # [B, 512, T', H', W']

        # Pool spatial dimensions, keep temporal
        features = self.temporal_pool(features)  # [B, 512, T', 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 512, T']
        features = features.permute(0, 2, 1)  # [B, T', 512]

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(features)  # [B, T', hidden*2]

        # Use both last hidden state and mean pooling
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # Forward and backward
        temporal_mean = lstm_out.mean(dim=1)

        # Combine both representations
        combined = final_hidden + temporal_mean

        # Predictions
        clip_pred = self.classifier(combined)
        delta_pred = self.volume_regressor(combined).squeeze(-1)

        return clip_pred, delta_pred

class VolumeSequenceModel(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()

        # Backbone: pretrained R2Plus1D (remove last 2 layers)
        full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(full_model.children())[:-2])  # output: [B, 512, T', H', W']
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep T', pool H/W

        # LSTM over sequence of clips
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Final regressor (predict total delta volume over the sequence)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: [B, S, C, T, H, W]
        B, S, C, T, H, W = x.shape
        x = x.view(B * S, C, T, H, W)  # → [B*S, C, T, H, W]

        # Backbone feature extraction
        feats = self.backbone(x)              # [B*S, 512, T', H', W']
        feats = self.spatial_pool(feats)      # [B*S, 512, T', 1, 1]
        feats = feats.squeeze(-1).squeeze(-1) # [B*S, 512, T']
        feats = feats.mean(dim=2)             # [B*S, 512]

        # Reshape to sequence
        feats = feats.view(B, S, 512)         # [B, S, 512]

        # LSTM across clips
        lstm_out, _ = self.lstm(feats)        # [B, S, hidden_dim * 2]
        seq_repr = lstm_out[:, -1, :]         # use last step

        # Predict cumulative volume over sequence
        pred_volume = self.regressor(seq_repr).squeeze(-1)  # [B]

        return pred_volume

class MultiTaskVolumeSequenceModel(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256, dropout=0.3):
        super().__init__()

        # Backbone: pretrained R2Plus1D (remove last 2 layers)
        full_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(full_model.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep T', pool H/W

        # LSTM over sequence of clips
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Clip-level classification head
        self.clip_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Sequence-level volume regressor
        self.volume_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.sequence_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, S, C, T, H, W]
        B, S, C, T, H, W = x.shape
        x = x.view(B * S, C, T, H, W)

        # Backbone feature extraction
        feats = self.backbone(x)              # [B*S, 512, T', H', W']
        feats = self.spatial_pool(feats)      # spatial pooling -> keep temporal
        feats = feats.squeeze(-1).squeeze(-1) # [B*S, 512, T']
        clip_features = feats.mean(dim=2)     # [B*S, 512] - averaged over time

        # Clip-level predictions
        clip_preds = self.clip_classifier(clip_features)  # [B*S, num_classes]
        clip_preds = clip_preds.view(B, S, -1)  # [B, S, num_classes]

        # Reshape to sequence for LSTM
        feats = clip_features.view(B, S, 512)  # [B, S, 512]

        # LSTM across clips
        lstm_out, _ = self.lstm(feats)
        # seq_repr = lstm_out.mean(dim=1) # mean pooling over sequence
        seq_repr = lstm_out[:, -1, :] # use last step [B, hidden_dim * 2]

        # Sequence-level predictions
        volume_pred = self.volume_regressor(seq_repr).squeeze(-1)  # [B]
        sequence_cls_pred = self.sequence_classifier(seq_repr)     # [B, num_classes]

        return clip_preds, volume_pred, sequence_cls_pred
