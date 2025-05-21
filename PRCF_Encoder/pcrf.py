# prcf.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Channel Fusion Module (SENet-like) ---
class ChannelFusionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        if in_channels <= reduction_ratio:
            reduced_channels = in_channels // 2 if in_channels > 1 else 1
        else:
            reduced_channels = in_channels // reduction_ratio
        if reduced_channels < 1: # Đảm bảo reduced_channels ít nhất là 1
            reduced_channels = 1

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale

# --- Receptive Field Block (RFB) - Basic Version ---
class BasicRFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4 if in_channels >= 4 else in_channels

        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=5, dilation=5)
        )
        # Đảm bảo số kênh đầu ra từ concat phù hợp
        self.conv_linear = nn.Conv2d(inter_channels * 3, out_channels, kernel_size=1)
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        x_shortcut = self.shortcut_conv(x) # Shortcut để cộng residual

        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        out = torch.cat((b0, b1, b2), dim=1)
        out = self.conv_linear(out)
        out = self.bn_final(out + x_shortcut)
        return self.relu_final(out)

# --- PRCF Encoder ---
class PRCFEncoder(nn.Module):
    def __init__(self, global_feat_channels, lesion_prompt_channels, out_channels_prcf, reduction_ratio_cfm=16):
        super().__init__()
        self.global_feat_channels = global_feat_channels
        self.lesion_prompt_channels = lesion_prompt_channels
        concatenated_channels = global_feat_channels + lesion_prompt_channels

        self.channel_fusion = ChannelFusionModule(concatenated_channels, reduction_ratio=reduction_ratio_cfm)
        self.receptive_module = BasicRFB(concatenated_channels, out_channels_prcf) # out_channels_prcf sẽ là số kênh đầu ra của RFB

        if global_feat_channels != out_channels_prcf:
            self.p_adapter = nn.Sequential(
                nn.Conv2d(global_feat_channels, out_channels_prcf, kernel_size=1),
                nn.BatchNorm2d(out_channels_prcf)
            )
        else:
            self.p_adapter = nn.Identity()

    def forward(self, P_global_features, G_lesion_prompt):
        prcf_device = P_global_features.device # Giả định P và G sẽ ở cùng device
        # print(f"--- PRCFEncoder Forward ---")
        # print(f"--- PRCF: P_global_features.device: {P_global_features.device}, shape: {P_global_features.shape} ---")
        # print(f"--- PRCF: G_lesion_prompt.device: {G_lesion_prompt.device}, shape: {G_lesion_prompt.shape} ---")

        # Di chuyển các sub-module của PRCF sang device nếu cần (an toàn hơn)
        self.channel_fusion.to(prcf_device)
        self.receptive_module.to(prcf_device)
        if hasattr(self, 'p_adapter'): self.p_adapter.to(prcf_device)


        if P_global_features.shape[2:] != G_lesion_prompt.shape[2:]:
            G_lesion_prompt_resized = F.interpolate(G_lesion_prompt,
                                           size=P_global_features.shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            print(f"--- PRCF: G_lesion_prompt_resized.device: {G_lesion_prompt_resized.device}, shape: {G_lesion_prompt_resized.shape} ---")
        else:
            G_lesion_prompt_resized = G_lesion_prompt

        F_concat = torch.cat((P_global_features, G_lesion_prompt_resized), dim=1)
        #print(f"--- PRCF: F_concat.device: {F_concat.device}, shape: {F_concat.shape} ---")

        F_fused_channel = self.channel_fusion(F_concat)
        #print(f"--- PRCF: F_fused_channel.device: {F_fused_channel.device}, shape: {F_fused_channel.shape} ---")

        F_receptive_out = self.receptive_module(F_fused_channel)
        #print(f"--- PRCF: F_receptive_out.device: {F_receptive_out.device}, shape: {F_receptive_out.shape} ---")

        P_adapted_for_residual = self.p_adapter(P_global_features)
        #print(f"--- PRCF: P_adapted_for_residual.device: {P_adapted_for_residual.device}, shape: {P_adapted_for_residual.shape} ---")

        F_final = F_receptive_out + P_adapted_for_residual
        F_final = F.relu(F_final)
        #print(f"--- PRCF: F_final (output).device: {F_final.device}, shape: {F_final.shape} ---")
        #print(f"--- PRCFEncoder Forward Complete ---")
        return F_final