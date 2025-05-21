# config.py (Ví dụ)
import torch

# Paths
DATA_DIR = 'E:\\EYEPACS'  
TRAIN_IMG_DIR = DATA_DIR + '\\train\\'
TEST_IMG_DIR = DATA_DIR + '\\test\\'
LABEL_FILE = DATA_DIR + '\\trainLabels_cleaned.csv'
MODEL_SAVE_PATH = 'models_weight\\pretrained_efficientnet_latest_best.pth.tar'

# Data parameters
IMG_SIZE = 224 # Kích thước ảnh đầu vào cho EfficientNet (có thể thay đổi)  
BATCH_SIZE = 16
VALID_SPLIT = 0.2 # Tỷ lệ dữ liệu dùng cho validation
NUM_WORKERS = 4 # Số luồng tải dữ liệu

# Model parameters
CNN_MODEL_NAME = 'efficientnet_b0' # Hoặc 'efficientnet_b3' như paper gợi ý nếu đủ tài nguyên
NUM_CLASSES =  1 # Hoặc 1 nếu dùng MSE loss như paper (cần xem xét kỹ)
# Nếu NUM_CLASSES = 1, dùng MSELoss. Nếu = 5, dùng CrossEntropyLoss (phổ biến hơn cho phân loại)
# Bài báo dùng MSE (Eq. 1), vậy ta sẽ dùng NUM_CLASSES = 1
OUTPUT_NEURONS = 1

# Pre-training parameters
PRETRAIN_EPOCHS = 50 # Số epochs để tiền huấn luyện CNN (cần nhiều hơn trong thực tế)
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grad-CAM parameters
TARGET_LAYER_NAME = 'features.8' # Tên lớp conv cuối trong EfficientNet 
# Để kiểm tra tên lớp:
# model = torchvision.models.efficientnet_b0()
# print(dict(model.named_modules()))
