# wsl_dr_model.py
import torch
import torch.nn as nn
import timm # For Swin Transformer
import numpy as np

import torch.nn.functional as F # Để sử dụng các hàm kích hoạt như ReLU
import models.cnn_model as cnn_prompt_generator # Model để tạo prompt (EfficientNet)
import attention_modules.attention_utils # Hàm create_prompt_from_image
from PRCF_Encoder.pcrf import PRCFEncoder # Import PRCFEncoder từ file prcf.py
from sklearn.metrics import cohen_kappa_score # Để tính Quadratic Weighted Kappa
import configs


class WSL_DR_Model(nn.Module):
    def __init__(self,
                 cnn_prompt_model_name=configs.CNN_MODEL_NAME,
                 cnn_prompt_checkpoint_path=configs.MODEL_SAVE_PATH,
                 grad_cam_target_layer=configs.TARGET_LAYER_NAME,
                 swin_model_name='swin_tiny_patch4_window7_224', # Ví dụ Swin-T
                 swin_pretrained=True,
                 num_dr_classes=configs.OUTPUT_NEURONS, # 1 cho MSE loss
                 prcf_reduction_ratio=16,
                 img_size=configs.IMG_SIZE): # Kích thước ảnh đầu vào cho Swin
        super().__init__()

        self.img_size = img_size
        self.grad_cam_target_layer = grad_cam_target_layer

        # 1. Attention Module (CNN để tạo Lesion Prompt)
        #    - Model này sẽ được load trọng số tiền huấn luyện và đóng băng.
        self.cnn_for_prompt = cnn_prompt_generator.get_cnn_model(
            model_name=cnn_prompt_model_name,
            num_outputs=1 # Luôn là 1 cho MSE loss của CNN tiền huấn luyện
        )
        self.cnn_for_prompt = self.cnn_for_prompt.to(configs.DEVICE) # Đưa model lên device
        # Load checkpoint cho CNN tạo prompt
        try:
            print(f"Loading checkpoint for CNN prompt generator from: {cnn_prompt_checkpoint_path}")
            checkpoint = torch.load(cnn_prompt_checkpoint_path, map_location='cpu') # Load lên CPU trước
            self.cnn_for_prompt.load_state_dict(checkpoint['state_dict'])
            print("CNN prompt generator loaded successfully.")
        except FileNotFoundError:
            print(f"WARNING: Checkpoint for CNN prompt generator not found at {cnn_prompt_checkpoint_path}. Using randomly initialized weights for prompt generation (NOT RECOMMENDED).")
        except Exception as e:
            print(f"ERROR loading CNN prompt generator checkpoint: {e}. Using randomly initialized weights.")

        # Đóng băng CNN tạo prompt
        for param in self.cnn_for_prompt.parameters():
            param.requires_grad = False
        self.cnn_for_prompt.eval() # Luôn ở chế độ eval

        # 2. Backbone chính (Swin Transformer)
        #    - Phần đầu của Swin sẽ dùng để trích xuất Global Features 'P'.
        #    - Phần sau sẽ là Classification Module.
        self.swin_transformer = timm.create_model(
            swin_model_name,
            pretrained=swin_pretrained,
            num_classes=0, # Loại bỏ lớp head mặc định, ta sẽ tự thêm
            features_only=False # Lấy features từ các stage khác nhau nếu cần features_only=True
        )

        # Xác định số kênh của Global Features 'P' từ Swin Transformer
        # 'P' sẽ là đầu ra sau patch_embed và có thể là stage đầu tiên.
        # Ví dụ, Swin-T tiny: patch_embed -> 96 channels
        # self.swin_transformer.patch_embed.proj.out_channels
        self.global_feature_channels = self.swin_transformer.patch_embed.proj.out_channels # Kênh sau patch_embed
        # Kích thước không gian của P (H_p, W_p)
        # Lấy grid_size từ patch_embed để đảm bảo tính nhất quán
        # self.swin_transformer.patch_embed.grid_size là một tuple (Hp, Wp)
        # Tính toán kích thước dự kiến từ img_size và patch_size để so sánh
        if isinstance(self.swin_transformer.patch_embed.patch_size, int):
            patch_size_val = self.swin_transformer.patch_embed.patch_size
            self.expected_P_H = img_size // patch_size_val
            self.expected_P_W = img_size // patch_size_val
        else:
            patch_size_h, patch_size_w = self.swin_transformer.patch_embed.patch_size
            self.expected_P_H = img_size // patch_size_h
            self.expected_P_W = img_size // patch_size_w        
            
        # self.P_feature_map_size không còn cần thiết nếu dùng P_grid_size_H, P_grid_size_W

        # 3. PRCF Encoder
        #    - global_feat_channels: từ Swin (ví dụ 96)
        #    - lesion_prompt_channels: 3 (RGB)
        #    - out_channels_prcf: nên bằng global_feat_channels để có thể đưa lại vào Swin
        self.prcf_encoder = PRCFEncoder(
            global_feat_channels=self.global_feature_channels,
            lesion_prompt_channels=3, # G là ảnh prompt RGB
            out_channels_prcf=self.global_feature_channels, # Output cùng số kênh với P
            reduction_ratio_cfm=prcf_reduction_ratio
        )

        # 4. Classification Module (Phần còn lại của Swin Transformer + Head mới)
        #    - Input cho phần này sẽ là output của PRCF Encoder.
        #    - Số kênh đầu vào cho các layer tiếp theo của Swin phải khớp.
        #    - Ta sẽ sử dụng các stages còn lại của Swin.
        #    - Đầu ra của Swin (sau các stages) sẽ là một vector đặc trưng.
        #    - Cuối cùng là một lớp Linear để phân loại DR.

        # Lấy số kênh đầu ra của Swin Transformer (trước lớp head mặc định)
        # self.num_features_swin_output = self.swin_transformer.num_features
        self.num_features_swin_output = self.swin_transformer.head.in_features # Lấy từ in_features của head cũ

        # Lớp head mới cho DR grading
        self.dr_head = nn.Linear(self.num_features_swin_output, num_dr_classes)


    def forward(self, x_original_image):
        """
        x_original_image: Batch ảnh đầu vào gốc, shape [B, 3, IMG_SIZE, IMG_SIZE]
                          (đã qua transform chuẩn hóa của dataset)
        """
        batch_size = x_original_image.shape[0]
        device = x_original_image.device

        # --- Bước A: Tạo Lesion Prompt G (sử dụng Attention Module) ---
        # Di chuyển cnn_for_prompt lên device nếu chưa

        if x_original_image.shape[2] != configs.IMG_SIZE: # Giả định IMG_SIZE của config là cho CNN prompt
             x_for_prompt_cnn = F.interpolate(x_original_image,
                                         size=(configs.IMG_SIZE, configs.IMG_SIZE),
                                         mode='bilinear', align_corners=False)
        else:
             x_for_prompt_cnn = x_original_image

        # Tạo prompt vẫn nằm trong ngữ cảnh chung, không cần no_grad() riêng ở đây
        
        with torch.set_grad_enabled(True):
            # Đảm bảo input cho create_prompt_from_image cũng có thể tạo grad nếu cần
            # create_prompt_from_image sẽ tự xử lý requires_grad của model và input bên trong nó
            G_lesion_prompt, _ = attention_modules.attention_utils.create_prompt_from_image(
                x_for_prompt_cnn, # Sẽ được clone và .requires_grad_(True) bên trong
                self.cnn_for_prompt,
                self.grad_cam_target_layer,
                device # Truyền device hiện tại
            )
        # G_lesion_prompt sẽ được tạo ra, nhưng không có gradient nào chảy ngược vào cnn_for_prompt
        # Tuy nhiên, G_lesion_prompt giờ đây sẽ là một phần của đồ thị tính toán chính
        # và gradient có thể chảy ngược *qua* nó (từ các module sau) nếu nó được sử dụng
        # trong các phép toán có thể tính gradient.

        # Để chắc chắn G_lesion_prompt không tự nó có gradient không cần thiết
        # và để ngắt kết nối nó khỏi đồ thị của cnn_for_prompt (mặc dù cnn_for_prompt đã đóng băng)
        # bạn có thể .detach() nó sau khi tạo ra, trước khi đưa vào PRCF.
        G_lesion_prompt = G_lesion_prompt.detach()

        # G_lesion_prompt shape: [B, 3, IMG_SIZE_CNN_PROMPT, IMG_SIZE_CNN_PROMPT]

        # --- Bước B: Qua patch_embed của Swin Transformer ---
        # x_after_embed shape: [B, L, C_embed], L = H_p * W_p
        if x_original_image.shape[2] != self.img_size or x_original_image.shape[3] != self.img_size: # Kiểm tra cả chiều cao và rộng
            print(f"Resizing input for Swin from {x_original_image.shape} to ({self.img_size}, {self.img_size})")
            x_for_swin = F.interpolate(x_original_image,
                                         size=(self.img_size, self.img_size),
                                         mode='bilinear', align_corners=False)
        else:
            # Nếu kích thước đã đúng, gán x_original_image cho x_for_swin
            x_for_swin = x_original_image

        x_after_embed = self.swin_transformer.patch_embed(x_for_swin)
        x_after_embed = self.swin_transformer.patch_embed(x_for_swin)
        # x_after_embed._input_HW = (H,W) # Một số phiên bản SwinV2 có thể có cái này

        # Lấy H_p, W_p từ patch_embed.grid_size
        H_p, W_p = self.swin_transformer.patch_embed.grid_size
        assert H_p == self.expected_P_H and W_p == self.expected_P_W, \
            f"Mismatch in H_p, W_p. From patch_embed: ({H_p},{W_p}), Expected from img_size/patch_size: ({self.expected_P_H},{self.expected_P_W})"

        # Reshape x_after_embed (P_global_features_flat) về dạng [B, C_embed, H_p, W_p] cho PRCF
        P_global_features_for_prcf = x_after_embed.transpose(1, 2).contiguous().view(
            batch_size, self.global_feature_channels, H_p, W_p
        )

        # --- Bước C: Đưa P (đã reshape) và G qua PRCF Encoder ---
        F_prcf_output_spatial = self.prcf_encoder(P_global_features_for_prcf, G_lesion_prompt)
        # F_prcf_output_spatial shape: [B, C_embed (out_channels_prcf), H_p, W_p]

        # --- Bước D: Đưa output của PRCF qua các STAGES của Swin Transformer ---
        # Reshape F_prcf_output_spatial về dạng [B, L, C_embed]
        # L = H_p * W_p
        x_modified_for_stages = F_prcf_output_spatial.permute(0, 2, 3, 1).contiguous()  # [B, H_p, W_p, C]

        # Cho x_modified_for_stages đi qua các stages (BasicLayer) của Swin Transformer
        # Mỗi BasicLayer trong self.swin_transformer.layers nhận input x (B,L,C)
        # và quan trọng là nó tự quản lý H, W bên trong dựa trên L và cấu hình của nó.

        current_x = x_modified_for_stages # Giả sử đây là input cho stage đầu tiên, dạng [B, L, C]
        for i, layer_module in enumerate(self.swin_transformer.layers):
            current_x = layer_module(current_x)
            # current_x sau mỗi stage có thể vẫn là [B, L_new, C_new]
            # Hoặc nó có thể được reshape lại thành [B, H_new, W_new, C_new] bởi lớp norm cuối cùng
            # của Swin Transformer hoặc cách timm trả về.

        # Output của stage cuối cùng (current_x) có thể cần được reshape lại nếu nó là [B,L,C]
        # trước khi vào self.swin_transformer.norm nếu norm đó mong đợi 4D.
        # Tuy nhiên, self.swin_transformer.norm thường áp dụng trên chiều cuối (C).

        # Hãy kiểm tra shape của current_x ngay sau vòng lặp các stage
        print(f"Shape of current_x after all Swin stages: {current_x.shape}")

        x_final_features = self.swin_transformer.norm(current_x)
        print(f"Shape after norm (x_final_features): {x_final_features.shape}")
        # Bây giờ chúng ta biết x_final_features là [B, H_f, W_f, C_f], ví dụ [8, 7, 7, 768]

        # Global Average Pooling trên cả chiều H_f và W_f
        # Nếu x_final_features là [B, H, W, C]
        # Chúng ta muốn lấy mean trên dim=1 (H) và dim=2 (W)
        if x_final_features.ndim == 4: # Đảm bảo nó là 4D
            # Cách 1: mean tuần tự
            # x_pooled = x_final_features.mean(dim=2)  # Mean qua W -> [B, H, C]
            # x_pooled = x_pooled.mean(dim=1)        # Mean qua H -> [B, C]

            # Cách 2: mean qua nhiều chiều cùng lúc (khuyến nghị)
            x_pooled = x_final_features.mean(dim=[1, 2]) # Mean qua cả H và W
                                                       # Kết quả là [B, C_f], ví dụ [8, 768]
        elif x_final_features.ndim == 3: # Trường hợp nó là [B, L, C]
            x_pooled = x_final_features.mean(dim=1) # Mean qua L -> [B, C]
        else:
            # Xử lý lỗi hoặc một trường hợp không mong đợi
            raise ValueError(f"Unexpected shape for x_final_features: {x_final_features.shape}")


        print(f"Shape before dr_head (x_pooled): {x_pooled.shape}") # Nên là [8, 768]

        dr_logits = self.dr_head(x_pooled)
        print(f"Shape after dr_head (dr_logits/scores): {dr_logits.shape}") # Nên là [8, 1]

        return dr_logits # [B, num_dr_classes]

def convert_outputs_to_classes(outputs, targets, num_classes=5, output_is_regression=True):
    """
    Chuyển đổi output của model (nếu là regression) và target thành các lớp rời rạc.
    Args:
        outputs: Tensor output từ model. Shape [B, 1] nếu regression, [B, num_classes] nếu classification.
        targets: Tensor target. Shape [B, 1] (giá trị float) hoặc [B] (nhãn lớp).
        num_classes: Số lượng lớp DR (thường là 5).
        output_is_regression: True nếu output của model là giá trị float (cần binning).
                              False nếu output là logits cho các lớp.
    Returns:
        predictions_cls: numpy array các lớp dự đoán.
        targets_cls: numpy array các lớp thực tế.
    """
    if output_is_regression:
        # Binning cho output regression
        # Đây là một ví dụ đơn giản, bạn có thể cần các ngưỡng (thresholds) tốt hơn
        # Các ngưỡng này nên được xác định dựa trên phân phối dữ liệu hoặc tối ưu hóa
        # thresholds = [0.5, 1.5, 2.5, 3.5] # Ngưỡng cho 5 lớp 0, 1, 2, 3, 4
        # Giả sử model output là giá trị liên tục, ta làm tròn về số nguyên gần nhất
        # và clip vào khoảng [0, num_classes-1]
        predictions_cls = torch.round(outputs.squeeze()).clamp(0, num_classes - 1).cpu().numpy().astype(int)
    else: # Output là logits cho các lớp
        predictions_cls = torch.argmax(outputs, dim=1).cpu().numpy().astype(int)

    # Chuyển đổi targets
    # Nếu targets là float (ví dụ từ MSE loss), cũng cần binning hoặc làm tròn
    if targets.dtype == torch.float32 or targets.dtype == torch.float64:
        targets_cls = torch.round(targets.squeeze()).clamp(0, num_classes - 1).cpu().numpy().astype(int)
    else: # Targets đã là nhãn lớp
        targets_cls = targets.cpu().numpy().astype(int)

    return predictions_cls, targets_cls

# --- Hàm huấn luyện và validation  ---
def train_wsl_dr_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    # Đảm bảo cnn_for_prompt luôn ở eval mode
    if hasattr(model, 'cnn_for_prompt'):
        model.cnn_for_prompt.eval()

    loop = tqdm(loader, leave=True, desc="Training WSL-DR")
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        if data is None or data.nelement() == 0: continue # Bỏ qua batch rỗng
        print(f"DEBUG train_epoch: Batch {batch_idx} - data original device: {data.device}")
        data, targets = data.to(device), targets.to(device)
        print(f"DEBUG train_epoch: Batch {batch_idx} - data moved to: {data.device}" )
        
        optimizer.zero_grad()
        scores = model(data)
        print(f"DEBUG train_epoch: Batch {batch_idx} - scores device: {scores.device}")
        #debug: printShape
        print(f"Batch {batch_idx}: scores.shape = {scores.shape}, targets.shape = {targets.shape}")

        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    print(f"WSL-DR Avg Training Loss: {avg_loss:.4f}")
    return avg_loss

# (Tương tự cho validate_wsl_dr_one_epoch)

def validate_wsl_dr_one_epoch(loader, model, loss_fn, device, num_dr_classes_for_kappa=5):
    model.eval() # Quan trọng!
    # Đảm bảo cnn_for_prompt cũng ở eval mode nếu có
    if hasattr(model, 'cnn_for_prompt'):
        model.cnn_for_prompt.eval()

    running_loss = 0.0
    all_targets_cls = []
    all_predictions_cls = []

    with torch.no_grad(): # Quan trọng cho validation!
        loop = tqdm(loader, leave=True, desc="Validating WSL-DR")
        for batch_idx, (data, targets) in enumerate(loop):
            print(f"DEBUG val_epoch: Batch {batch_idx} - data original device: {data.device}")
            if data is None or data.nelement() == 0: continue
            data, targets = data.to(device), targets.to(device)
            print(f"DEBUG val_epoch: Batch {batch_idx} - data moved to: {data.device}")


            scores = model(data) # Output của model, shape [B, 1] nếu num_dr_classes=1
            print(f"DEBUG val_epoch: Batch {batch_idx} - scores device: {scores.device}")
            loss = loss_fn(scores, targets)
            running_loss += loss.item()

            # Chuyển đổi scores (float) và targets (float) thành các lớp rời rạc (int) để tính Kappa
            # Giả sử targets là [B, 1] chứa các nhãn float 0.0, 1.0, ..., 4.0
            # Giả sử scores là [B, 1] chứa các dự đoán float

            # Xử lý targets:
            # .squeeze() để bỏ chiều cuối cùng nếu có, sau đó .round().long()
            # targets_cpu = targets.cpu().squeeze().round().long().numpy() # Chuyển về numpy array int
            # Hoặc giữ trên tensor:
            batch_targets_cls_tensor = targets.squeeze(dim=-1).round().long() # Bỏ dim 1, làm tròn, chuyển sang long
            all_targets_cls.extend(batch_targets_cls_tensor.cpu().tolist()) # Chuyển sang list Python

            # Xử lý predictions (scores):
            # scores có thể là giá trị âm hoặc lớn hơn num_dr_classes_for_kappa - 1
            # Cần clip giá trị dự đoán vào khoảng [0, num_dr_classes_for_kappa - 1]
            # predictions_cpu = scores.cpu().squeeze().round().long()
            # predictions_cpu = torch.clamp(predictions_cpu, 0, num_dr_classes_for_kappa - 1).numpy()
            # Hoặc giữ trên tensor:
            batch_preds_cls_tensor = scores.squeeze(dim=-1).round().long()
            batch_preds_cls_tensor = torch.clamp(batch_preds_cls_tensor, 0, num_dr_classes_for_kappa - 1)

            # --- ĐÂY LÀ NƠI CẦN SỬA ---
            # Đảm bảo batch_preds_cls_tensor là một iterable (1D tensor hoặc list)
            # Ngay cả khi batch_size=1, .tolist() sẽ tạo ra một list chứa 1 phần tử
            # Nếu batch_preds_cls_tensor là 0-D (khi batch_size=1 và squeeze hoàn toàn),
            # .tolist() sẽ gây lỗi hoặc trả về số Python.
            # Cách an toàn hơn là đảm bảo nó là 1D tensor trước khi tolist()

            if batch_preds_cls_tensor.ndim == 0: # Nếu là scalar (0-D tensor)
                # Chuyển nó thành list chứa một phần tử
                list_to_extend = [batch_preds_cls_tensor.item()]
            else: # Nếu là 1-D tensor hoặc cao hơn (dù thường là 1D ở đây)
                list_to_extend = batch_preds_cls_tensor.cpu().tolist()

            all_predictions_cls.extend(list_to_extend)
            # --------------------------

            loop.set_postfix(val_loss=loss.item())

    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    print(f"WSL-DR Avg Validation Loss: {avg_loss:.4f}")

    # Tính Quadratic Weighted Kappa
    if len(all_targets_cls) > 0 and len(all_predictions_cls) > 0:
        # Chuyển đổi all_targets_cls và all_predictions_cls thành numpy arrays nếu chưa
        targets_np = np.array(all_targets_cls)
        predictions_np = np.array(all_predictions_cls)
        kappa = cohen_kappa_score(targets_np, predictions_np, weights='quadratic')
        print(f"Validation Quadratic Weighted Kappa: {kappa:.4f}")
    else:
        print("Not enough data to calculate Kappa.")
        kappa = 0.0 # Hoặc None

    return avg_loss, kappa



def check_device_consistency(model: torch.nn.Module):
    """
    Kiểm tra xem tất cả parameters và buffers của model đã nằm trên cùng một device chưa.
    In ra cảnh báo nếu phát hiện tensor khác device.
    """
    devices_found = set()

    for name, param in model.named_parameters():
        devices_found.add(param.device)

    for name, buf in model.named_buffers():
        devices_found.add(buf.device)

    if len(devices_found) > 1:
        print("[WARNING] Multiple devices detected in model:")
        for device in devices_found:
            print(" -", device)
        print("You should move the entire model to a single device using model.to(device)")
    else:
        print(f"[OK] All model tensors are on a single device: {devices_found.pop()}")



# Ví dụ cách sử dụng và huấn luyện
if __name__ == '__main__':
    from tqdm import tqdm
    import prepare_data.data_loader as main_data_loader
    import torch.optim as optim
    import configs.utils # Giả sử bạn có utils.py để lưu checkpoint

    global_device_setting = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    print("Initializing WSL-DR Model...")
    wsl_model = WSL_DR_Model(
        cnn_prompt_model_name='efficientnet_b0',
        cnn_prompt_checkpoint_path=configs.MODEL_SAVE_PATH,
        grad_cam_target_layer=configs.TARGET_LAYER_NAME,
        swin_model_name='swin_tiny_patch4_window7_224',
        swin_pretrained=True,
        num_dr_classes=configs.OUTPUT_NEURONS,
        img_size=224
    ).to(global_device_setting)

    print(f"DEBUG main: wsl_model moved to {next(wsl_model.parameters()).device if list(wsl_model.parameters()) else 'N/A'}")
    # Kiểm tra xem model đã nằm trên device chưa
    check_device_consistency(wsl_model)
    print("Model initialized successfully.")
    
    print("Loading data for WSL-DR training...")
    train_loader, valid_loader = main_data_loader.get_dataloaders(
        configs.LABEL_FILE,
        configs.TRAIN_IMG_DIR,
        configs.BATCH_SIZE // 2 if torch.cuda.is_available() else configs.BATCH_SIZE // 4, # Điều chỉnh batch size
        configs.VALID_SPLIT,
        wsl_model.img_size,
        configs.NUM_WORKERS
    )

    if configs.OUTPUT_NEURONS == 1:
        loss_fn_wsl = nn.MSELoss()
        print("Using MSELoss for WSL-DR training.")
    else:
        loss_fn_wsl = nn.CrossEntropyLoss() # Nếu num_dr_classes > 1
        print(f"Using CrossEntropyLoss for WSL-DR training with {configs.OUTPUT_NEURONS} classes.")


    optimizer_wsl = optim.AdamW(
        filter(lambda p: p.requires_grad, wsl_model.parameters()),
        lr=1e-5, # Bắt đầu với learning rate nhỏ
        weight_decay=0.01
    )
    # Cân nhắc sử dụng scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_wsl, 'min', patience=3, factor=0.5)


    num_epochs_wsl = 50 # Tăng số epochs
    best_val_kappa = -1.0 # Theo dõi Kappa tốt nhất
    best_val_loss = float('inf')

    print("Starting WSL-DR training...")
    for epoch in range(num_epochs_wsl):
        print(f"\n--- WSL-DR Epoch {epoch+1}/{num_epochs_wsl} ---")

        # Huấn luyện
        train_loss = train_wsl_dr_one_epoch(train_loader, wsl_model, optimizer_wsl, loss_fn_wsl, global_device_setting)

        # Validate
        val_loss, val_kappa = validate_wsl_dr_one_epoch(
            valid_loader,
            wsl_model,
            loss_fn_wsl,
            global_device_setting   ,
            num_dr_classes_for_kappa=5 # Giả sử có 5 lớp DR (0-4)
        )

        # Cập nhật learning rate dựa trên val_loss
        scheduler.step(val_loss)


        # Lưu checkpoint dựa trên Kappa hoặc Loss
        # Ở đây ví dụ lưu dựa trên Kappa
        if val_kappa > best_val_kappa:
            print(f"Validation QWK improved ({best_val_kappa:.4f} --> {val_kappa:.4f}). Saving model...")
            best_val_kappa = val_kappa
            # Tạo một tên file checkpoint động
            checkpoint_filename = f"wsl_dr_epoch_{epoch+1}_qwk_{val_kappa:.4f}.pth.tar"
            if hasattr(configs.utils, 'save_checkpoint'): # Kiểm tra nếu utils.save_checkpoint tồn tại
                configs.utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': wsl_model.state_dict(),
                    'optimizer': optimizer_wsl.state_dict(),
                    'best_val_kappa': best_val_kappa,
                    'val_loss': val_loss
                }, filename=checkpoint_filename)
            else:
                print(f"utils.save_checkpoint not found. Skipping save for {checkpoint_filename}")
        elif val_loss < best_val_loss and val_kappa == -1: # Nếu không tính được kappa, dùng loss
             print(f"Validation Loss improved ({best_val_loss:.4f} --> {val_loss:.4f}) while QWK not computed. Saving model...")
             best_val_loss = val_loss


    print("WSL-DR training finished.")