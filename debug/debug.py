# Giả sử các import cần thiết đã có:
import torch
import torch.nn.functional as F
import configs.configs as configs
from attention_modules.grad_cam import GradCAM
import models.cnn_model as cnn_model
import configs.utils as utils
import prepare_data.data_loader # Để lấy ảnh mẫu
import matplotlib.pyplot as plt
import numpy as np
from attention_modules.attention_utils import normalize_heatmap_min_max, generate_lesion_prompt
import cv2

# --- Hàm debug từng bước ---
def debug_attention_module_step_by_step(image_tensor_single, cnn_model_instance, target_layer_name, device):
    """
    Debug từng bước việc tạo Lesion Prompt cho MỘT ảnh.
    Input:
        - image_tensor_single: Tensor ảnh ĐƠN LẺ đầu vào [C, H, W], đã qua transform.
                               KHÔNG PHẢI BATCH.
        - cnn_model_instance: Mô hình CNN đã được tiền huấn luyện.
        - target_layer_name: Tên lớp conv mục tiêu cho Grad-CAM.
        - device: Thiết bị (cpu/cuda).
    """
    print("--- BẮT ĐẦU DEBUG MODULE CHÚ Ý ---")

    # Chuẩn bị model và ảnh
    cnn_model_instance.eval()
    cnn_model_instance.to(device)
    image_batch_for_gradcam = image_tensor_single.unsqueeze(0).to(device) # GradCAM nhận batch

    # === BƯỚC 1: Tạo Heatmap thô bằng Grad-CAM ===
    print("\n[BƯỚC 1]: Tạo Heatmap thô bằng Grad-CAM")
    gradcam_obj = GradCAM(cnn_model_instance, target_layer_name)
    # Vì model dùng MSE (output=1), không cần target_class_index
    # Hàm __call__ của GradCAM đã bao gồm việc resize về kích thước ảnh gốc
    # và chuẩn hóa [0,1] cho từng ảnh trong batch
    raw_heatmap_batch = gradcam_obj(image_batch_for_gradcam, target_class_index=None)
    raw_heatmap_single = raw_heatmap_batch[0] # Lấy heatmap của ảnh duy nhất trong batch
    print(f"  - Kích thước heatmap thô (sau GradCAM): {raw_heatmap_single.shape}")
    print(f"  - Giá trị min/max heatmap thô: {raw_heatmap_single.min().item():.4f} / {raw_heatmap_single.max().item():.4f}")

    # Hiển thị heatmap thô (đã được resize và chuẩn hóa [0,1] bởi GradCAM)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    img_display = image_tensor_single.cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406]) # Giả sử chuẩn hóa theo ImageNet
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    plt.imshow(img_display)
    plt.title("Ảnh gốc (Un-normalized)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(raw_heatmap_single.cpu().numpy(), cmap='jet')
    plt.title("Heatmap thô từ Grad-CAM")
    plt.axis('off')
    # utils.visualize_heatmap(image_tensor_single, raw_heatmap_single) # Hoặc dùng hàm tiện ích

    # === BƯỚC 2: Chuẩn hóa Heatmap bằng Min-Max (Eq. 4) ===
    # Bước này có thể là lặp lại nếu GradCAM đã chuẩn hóa, nhưng để đảm bảo theo paper.
    print("\n[BƯỚC 2]: Chuẩn hóa Heatmap bằng Min-Max (Eq. 4)")
    # Hàm normalize_heatmap_min_max nhận tensor [H,W] hoặc [B,H,W]
    normalized_heatmap_single = normalize_heatmap_min_max(raw_heatmap_single)
    print(f"  - Kích thước heatmap đã chuẩn hóa: {normalized_heatmap_single.shape}")
    print(f"  - Giá trị min/max heatmap đã chuẩn hóa: {normalized_heatmap_single.min().item():.4f} / {normalized_heatmap_single.max().item():.4f}")

    # Hiển thị heatmap đã chuẩn hóa
    plt.subplot(1, 3, 3)
    plt.imshow(normalized_heatmap_single.cpu().numpy(), cmap='jet')
    plt.title("Heatmap chuẩn hóa Min-Max")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # === BƯỚC 3: Tạo "Weight Map" (Chính là normalized_heatmap_single) ===
    # Theo paper, "Weight Map V" là kết quả sau khi chuẩn hóa heatmap.
    print("\n[BƯỚC 3]: Tạo Weight Map V (chính là heatmap đã chuẩn hóa)")
    weight_map_V = normalized_heatmap_single # Shape [H, W]
    print(f"  - Kích thước Weight Map V: {weight_map_V.shape}")

    # === BƯỚC 4: Tạo Lesion Prompt (Eq. 5) ===
    # gi = Pi * vi (Pi là ảnh gốc, vi là weight map)
    print("\n[BƯỚC 4]: Tạo Lesion Prompt G")
    # Hàm generate_lesion_prompt nhận batch ảnh và batch weight map
    # Chúng ta cần unsqueeze ảnh đơn lẻ và weight map đơn lẻ thành batch size 1
    image_batch_for_prompt = image_tensor_single.unsqueeze(0).to(device) # [1, C, H, W]
    weight_map_batch_for_prompt = weight_map_V.unsqueeze(0).to(device)   # [1, H, W]

    lesion_prompt_batch = generate_lesion_prompt(image_batch_for_prompt, weight_map_batch_for_prompt)
    lesion_prompt_single = lesion_prompt_batch[0] # Lấy prompt của ảnh duy nhất
    print(f"  - Kích thước Lesion Prompt G: {lesion_prompt_single.shape}")
    print(f"  - Giá trị min/max Lesion Prompt G (kênh 0): {lesion_prompt_single[0].min().item():.4f} / {lesion_prompt_single[0].max().item():.4f}")


    # Hiển thị Lesion Prompt
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_display) # Ảnh gốc un-normalized
    plt.title("Ảnh gốc (Un-normalized)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    prompt_display_debug = lesion_prompt_single.cpu().permute(1, 2, 0).numpy()
    # Ảnh gốc đã chuẩn hóa (image_tensor_single) nên khi nhân với weight_map (0-1)
    # thì prompt_display_debug cũng đang ở thang chuẩn hóa.
    # Để hiển thị đúng màu, cần un-normalize nó giống như ảnh gốc.
    prompt_display_debug_unnorm = std * prompt_display_debug + mean
    prompt_display_debug_unnorm = np.clip(prompt_display_debug_unnorm, 0, 1)
    plt.imshow(prompt_display_debug_unnorm)
    plt.title("Lesion Prompt G (Un-normalized)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("\n--- KẾT THÚC DEBUG MODULE CHÚ Ý ---")
    return lesion_prompt_single, normalized_heatmap_single

# --- Ví dụ cách sử dụng hàm debug ---
if __name__ == "__main__":
    # 1. Load model đã tiền huấn luyện
    print("Loading pre-trained model...")
    loaded_model = cnn_model.get_cnn_model().to(configs.DEVICE) # Sửa lại tên biến
    try:
        utils.load_checkpoint(configs.MODEL_SAVE_PATH, loaded_model)
    except FileNotFoundError:
        print(f"ERROR: Pre-trained model not found at {configs.MODEL_SAVE_PATH}")
        print("Please run pretrain_cnn.py first.")
        exit()
    loaded_model.eval()

    # 2. Lấy một ảnh mẫu (ví dụ từ validation loader)
    print("Loading sample data...")
    _, val_loader = prepare_data.data_loader.get_dataloaders(
        configs.LABEL_FILE,
        configs.TRAIN_IMG_DIR,
        1, # Batch size 1 để dễ debug
        configs.VALID_SPLIT,
        configs.IMG_SIZE,
        0 # num_workers=0 để tránh lỗi pickling khi debug
    )
    try:
        sample_images, sample_labels = next(iter(val_loader))
        if sample_images is None or sample_images.nelement() == 0 :
            print("Could not load sample batch from val_loader.")
            exit()
    except StopIteration:
        print("Validation loader is empty. Make sure you have data and valid_split is appropriate.")
        exit()


    # Chọn 1 ảnh từ batch (batch size đã là 1)
    # image_to_debug là tensor [C, H, W]
    image_to_debug = sample_images[0]
    print(f"Image to debug shape: {image_to_debug.shape}")


    # 3. Chạy hàm debug từng bước
    lesion_prompt_result, heatmap_result = debug_attention_module_step_by_step(
        image_to_debug,
        loaded_model,
        configs.TARGET_LAYER_NAME,
        configs.DEVICE
    )

    print("\nKết quả cuối cùng từ debug:")
    print(f"Lesion Prompt shape: {lesion_prompt_result.shape}")
    print(f"Heatmap shape: {heatmap_result.shape}")