import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 # Sử dụng OpenCV
import configs as configs

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    try:
        # Đảm bảo load về đúng device (CPU hoặc GPU)
        checkpoint = torch.load(checkpoint_path, map_location=configs.DEVICE)

        # Load model state dict
        # Xử lý trường hợp model được lưu với DataParallel (có tiền tố 'module.')
        state_dict = checkpoint['state_dict']
        if all(key.startswith('module.') for key in state_dict):
             print("=> Checkpoint saved with DataParallel, removing 'module.' prefix...")
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in state_dict.items():
                 name = k[7:] # remove `module.`
                 new_state_dict[name] = v
             model.load_state_dict(new_state_dict)
        else:
             model.load_state_dict(state_dict)

        # Load optimizer state dict (nếu có)
        if optimizer and 'optimizer' in checkpoint:
            try:
                 optimizer.load_state_dict(checkpoint['optimizer'])
                 print("=> Loaded optimizer state")
            except Exception as e:
                 print(f"Warning: Could not load optimizer state. Starting optimizer from scratch. Error: {e}")
        elif optimizer:
             print("Warning: Optimizer state not found in checkpoint. Starting optimizer from scratch.")

        # Lấy thông tin epoch và best_val_loss
        start_epoch = checkpoint.get('epoch', 0) # Lấy epoch, mặc định là 0 nếu không có
        best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Lấy loss, mặc định là inf

        print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, best_val_loss {best_val_loss:.4f})")
        return start_epoch, best_val_loss

    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return 0, float('inf') # Trả về giá trị mặc định nếu không tìm thấy file
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint from {checkpoint_path}. Error: {e}")
        return 0, float('inf') # Trả về giá trị mặc định nếu có lỗi khác

def visualize_heatmap(original_img_tensor, heatmap_tensor, alpha=0.6):
    """Hiển thị heatmap chồng lên ảnh gốc."""
    if heatmap_tensor.ndim == 3 and heatmap_tensor.shape[0] == 1:
        heatmap = heatmap_tensor.squeeze().cpu().numpy()
    elif heatmap_tensor.ndim == 2:
        heatmap = heatmap_tensor.cpu().numpy()
    else:
        raise ValueError("Heatmap tensor shape not supported")

    # Chuẩn hóa heatmap về [0, 255] và chuyển sang kiểu uint8
    heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Áp dụng colormap

    # Chuyển ảnh gốc từ tensor sang numpy (HxWxC) và đúng định dạng màu
    img = original_img_tensor.cpu().permute(1, 2, 0).numpy()
    # Nếu ảnh đã chuẩn hóa, cần un-normalize để hiển thị đúng
    # Giả sử chuẩn hóa theo ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img_display = np.uint8(img * 255)
    # Chuyển từ RGB (PyTorch/PIL) sang BGR (OpenCV) nếu cần
    img_display_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

    # Resize heatmap_color về kích thước ảnh gốc
    heatmap_resized = cv2.resize(heatmap_color, (img_display_bgr.shape[1], img_display_bgr.shape[0]))

    # Chồng heatmap lên ảnh gốc
    superimposed_img = cv2.addWeighted(img_display_bgr, 1 - alpha, heatmap_resized, alpha, 0)

    # Hiển thị (ví dụ)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_display_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('Superimposed')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return superimposed_img