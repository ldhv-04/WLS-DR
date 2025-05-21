# grad_cam.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # Đăng ký hook để lấy feature map và gradient
        self._register_hooks()

    def _hook_features(self, module, input, output):
        self.feature_maps = output.detach() # Lưu lại feature map

    def _hook_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach() # Lưu lại gradient

    def _register_hooks(self):
        # Tìm module tương ứng với target_layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model.")

        # Đăng ký forward và backward hook
        self.forward_handle = target_module.register_forward_hook(self._hook_features)
        self.backward_handle = target_module.register_full_backward_hook(self._hook_gradients) # Dùng full_backward_hook

    def _remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, target_class_index=None):
        """Tạo heatmap Grad-CAM."""
        self.model.eval() # Chuyển model sang evaluation mode

        # Đảm bảo input_tensor yêu cầu gradient
        input_tensor.requires_grad_(True)

        # 1. Forward pass để lấy output và feature maps (qua hook)
        output = self.model(input_tensor) # Shape [batch, num_classes] hoặc [batch, 1]

        # 2. Xác định điểm số mục tiêu
        if target_class_index is None:
            # Nếu output là [batch, 1] (regression/MSE), dùng chính output đó
            # Nếu output là [batch, C], dùng lớp có điểm số cao nhất
            if output.shape[1] == 1:
                target_score = output[:, 0]
            else:
                 target_score = output.max(1)[0] # Lấy điểm số max
                 # target_score = output[:, target_class_index] # Hoặc dùng index cụ thể
        else:
             target_score = output[:, target_class_index]

        # 3. Backward pass để lấy gradients (qua hook)
        self.model.zero_grad()
        # Tính tổng điểm số mục tiêu trên batch để backward một lần
        target_score_agg = target_score.sum()
        target_score_agg.backward(retain_graph=True) # Giữ lại graph nếu cần backward nhiều lần

        # Kiểm tra gradients và feature_maps đã được lưu chưa
        if self.gradients is None or self.feature_maps is None:
             self._remove_hooks() # Gỡ hook trước khi raise lỗi
             raise RuntimeError("Gradients or feature maps not captured.")

        # 4. Tính trọng số alpha_k (neuron importance weights) - Eq. 2 (simplified)
        # Global Average Pooling gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # Shape: [batch, channels, 1, 1]

        # 5. Tính tổ hợp tuyến tính có trọng số của feature maps - Eq. 3 (phần bên trong ReLU)
        # weights shape: [B, C, 1, 1], feature_maps shape: [B, C, H, W]
        weighted_features = weights * self.feature_maps # Broadcasting
        cam = torch.sum(weighted_features, dim=1) # Sum qua channels -> [B, H, W]

        # 6. Áp dụng ReLU
        cam = F.relu(cam)

        # 7. Resize CAM về kích thước ảnh gốc (ví dụ: nội suy bilinear)
        # Lấy kích thước ảnh gốc từ input_tensor
        original_height, original_width = input_tensor.shape[2:]
        cam_resized = F.interpolate(cam.unsqueeze(1), # Thêm dim channel -> [B, 1, H, W]
                                   size=(original_height, original_width),
                                   mode='bilinear',
                                   align_corners=False)
        cam_resized = cam_resized.squeeze(1) # Bỏ dim channel -> [B, H, W]

        # 8. (Optional) Chuẩn hóa CAM từng ảnh trong batch về [0, 1]
        cam_normalized = []
        for i in range(cam_resized.shape[0]):
            cam_img = cam_resized[i]
            cam_min = torch.min(cam_img)
            cam_max = torch.max(cam_img)
            if cam_max > cam_min:
                 cam_norm = (cam_img - cam_min) / (cam_max - cam_min)
            else:
                 cam_norm = torch.zeros_like(cam_img) # Tránh chia cho 0
            cam_normalized.append(cam_norm)
        cam_final = torch.stack(cam_normalized)

        # Gỡ bỏ hooks sau khi hoàn thành
        self._remove_hooks()

        return cam_final # Trả về heatmap đã chuẩn hóa [0, 1] cho batch