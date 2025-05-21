# cnn_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import configs.configs as cfg

def get_cnn_model(model_name=cfg.CNN_MODEL_NAME, num_outputs=cfg.OUTPUT_NEURONS, pretrained=True):
    """Lấy mô hình EfficientNet và thay đổi lớp classifier."""
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        # Thay thế lớp classifier cuối cùng
        model.classifier = nn.Linear(num_ftrs, num_outputs)
    elif model_name == 'efficientnet_b3':
         model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
         num_ftrs = model.classifier[1].in_features
         model.classifier = nn.Linear(num_ftrs, num_outputs)
    # Thêm các model khác nếu cần
    else:
        raise ValueError(f"Model {model_name} not supported yet.")

    # Nếu num_outputs > 1 (ví dụ: phân loại 5 lớp), không cần sigmoid ở đây
    # Nếu num_outputs = 1 (cho MSE loss), cũng không cần sigmoid vì MSE tính trên raw output

    return model
if __name__ == "__main__":
    #model = get_cnn_model()
   
    # Kiểm tra tên các lớp để xác định target_layer cho Grad-CAM
    #print(dict(model.named_modules())) # Tìm lớp conv cuối, ví dụ '_conv_head'
    # inp = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    # out = model(inp)
    # print("Output shape:", out.shape) # Phải là [2, 1] nếu OUTPUT_NEURONS=1
    # Trong cnn_model.py hoặc khi debug
    model = get_cnn_model()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d): # Hoặc một loại block cụ thể
            print(name)