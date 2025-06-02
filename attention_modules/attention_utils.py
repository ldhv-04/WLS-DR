# attention_utils.py
import sys
import os # Thêm os để kiểm tra file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import *   # Import các cấu hình từ file configs.pys
import torch
import torch.nn.functional as F
from grad_cam import GradCAM # Import lớp GradCAM vừa tạo
import models.cnn_model as cnn_model # Để load model
import configs.utils as utils # Để load checkpoint và visualize
import matplotlib.pyplot as plt
import numpy as np
import prepare_data.data_loader # Để lấy dữ liệu mẫu


def normalize_heatmap_min_max(heatmap):
    """Chuẩn hóa heatmap về [0, 1] dùng Min-Max (Eq. 4).
       Input: heatmap tensor [H, W] hoặc [B, H, W].
       Output: heatmap đã chuẩn hóa [0, 1].
    """
    if heatmap.ndim == 2: # Xử lý 1 heatmap
        heatmap_min = torch.min(heatmap)
        heatmap_max = torch.max(heatmap)
        if heatmap_max > heatmap_min:
            normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8) # Thêm epsilon tránh chia 0
        else:
            normalized = torch.zeros_like(heatmap)
        return normalized
    elif heatmap.ndim == 3: # Xử lý batch heatmap
        normalized_batch = []
        for i in range(heatmap.shape[0]):
            h_map = heatmap[i]
            h_min = torch.min(h_map)
            h_max = torch.max(h_map)
            if h_max > h_min:
                 norm_h = (h_map - h_min) / (h_max - h_min + 1e-8)
            else:
                 norm_h = torch.zeros_like(h_map)
            normalized_batch.append(norm_h)
        return torch.stack(normalized_batch)
    else:
         raise ValueError("Input heatmap must be 2D or 3D tensor.")


def generate_lesion_prompt(original_image_batch, weight_map_batch):
    """Tạo Lesion Prompt bằng phép nhân Hadamard (Eq. 5).
       Input:
         - original_image_batch: Tensor ảnh gốc [B, C, H, W]
         - weight_map_batch: Tensor bản đồ trọng số đã chuẩn hóa [B, H, W]
       Output: Lesion Prompt tensor [B, C, H, W]
    """
    # Đảm bảo weight_map có thêm chiều kênh để nhân với ảnh [B, H, W] -> [B, 1, H, W]
    if weight_map_batch.ndim == 3:
        weight_map_batch = weight_map_batch.unsqueeze(1)

    # Nhân element-wise (Hadamard product), weight_map sẽ được broadcast qua các kênh màu
    lesion_prompt = original_image_batch * weight_map_batch
    return lesion_prompt

# --- Hàm tổng hợp để tạo prompt từ ảnh và model ---
def create_prompt_from_image(image_tensor_batch, model_prompt_generator, target_layer_name, current_device):
    """
    Hàm tổng hợp các bước: chạy Grad-CAM, chuẩn hóa, tạo prompt.
    """
    # --- BƯỚC QUAN TRỌNG: Tạm thời bật requires_grad cho model tạo prompt ---
    # Lưu trạng thái requires_grad ban đầu
    model_prompt_generator.eval()
    model_prompt_generator.to(current_device)
    image_tensor_batch_on_device = image_tensor_batch.to(current_device)

    # input_for_gradcam đã được clone().detach().requires_grad_(True)
    input_for_gradcam = image_tensor_batch_on_device.clone().detach().requires_grad_(True)

    # Lưu trạng thái requires_grad của params model_prompt_generator
    original_param_grad_states = {}
    for name, param in model_prompt_generator.named_parameters():
        original_param_grad_states[name] = param.requires_grad
        param.requires_grad_(True) # Tạm thời bật grad

    try:
        gradcam = GradCAM(model_prompt_generator, target_layer_name)
        raw_heatmap_batch = gradcam(input_for_gradcam, target_class_index=None)
    finally:
        # Khôi phục trạng thái requires_grad
        for name, param in model_prompt_generator.named_parameters():
            if name in original_param_grad_states:
                param.requires_grad_(original_param_grad_states[name])

    normalized_heatmap_batch = normalize_heatmap_min_max(raw_heatmap_batch.detach())
    lesion_prompt_batch = generate_lesion_prompt(image_tensor_batch_on_device.detach(), # Sử dụng image_tensor_batch_on_device đã detach
                                              normalized_heatmap_batch)
    return lesion_prompt_batch, normalized_heatmap_batch

#Ví dụ cách sử dụng (trong một file script riêng, ví dụ: generate_prompts_example.py)
if __name__ == "__main__":
    # 1. Load model đã tiền huấn luyện
    print("Loading pre-trained model...")
    model = cnn_model.get_cnn_model().to(configs.DEVICE)
    try:
        utils.load_checkpoint(configs.MODEL_SAVE_PATH, model)
    except FileNotFoundError:
        print(f"ERROR: Pre-trained model not found at {configs.MODEL_SAVE_PATH}")
        print("Please run pretrain_cnn.py first.")
        exit()
    model.eval()

    # 2. Lấy một batch dữ liệu mẫu (ví dụ từ validation loader)
    print("Loading sample data...")
    _, val_loader = prepare_data.data_loader.get_dataloaders(
        configs.LABEL_FILE,
        configs.TRAIN_IMG_DIR,
        configs.BATCH_SIZE,
        configs.VALID_SPLIT,
        configs.IMG_SIZE,
        configs.NUM_WORKERS
    )
    sample_batch = next(iter(val_loader))
    images, labels = sample_batch
    if images is None or images.nelement() == 0:
         print("Could not load sample batch.")
         exit()

    # Chọn 1 ảnh từ batch để demo
    image_to_process = images[3].unsqueeze(0) # Lấy ảnh đầu tiên và giữ nguyên batch dim [1, C, H, W]
    print(f"Processing image with shape: {image_to_process.shape}")

    # 3. Tạo Lesion Prompt
    print("Generating Lesion Prompt...")
    lesion_prompt, normalized_heatmap = create_prompt_from_image(
        image_to_process,
        model,
        configs.TARGET_LAYER_NAME,
        configs.DEVICE
    )
    print(f"Generated Lesion Prompt shape: {lesion_prompt.shape}")
    print(f"Generated Normalized Heatmap shape: {normalized_heatmap.shape}")

    # 4. Visualize kết quả cho ảnh đầu tiên trong batch demo
    print("Visualizing results...")
    # Lấy ảnh gốc (cần un-normalize) và heatmap/prompt từ batch size 1
    original_image_to_vis = image_to_process[0] # Shape [C, H, W]
    heatmap_to_vis = normalized_heatmap[0]    # Shape [H, W]
    prompt_to_vis = lesion_prompt[0]          # Shape [C, H, W]

    # Hiển thị heatmap chồng lên ảnh gốc
    utils.visualize_heatmap(original_image_to_vis, heatmap_to_vis)

    # Hiển thị Lesion Prompt (cũng cần un-normalize nếu muốn xem màu sắc đúng)
    prompt_display = prompt_to_vis.cpu().permute(1, 2, 0).detach().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    prompt_display = std * prompt_display + mean # Có thể không cần un-normalize prompt
    prompt_display = np.clip(prompt_display, 0, 1)

    plt.figure()
    plt.imshow(prompt_display)
    plt.title('Generated Lesion Prompt')
    plt.axis('off')
    plt.show()