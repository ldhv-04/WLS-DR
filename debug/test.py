import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    # Tạo model đơn giản
    model = torch.nn.Linear(1000, 1000).to(device)
    # Tạo dữ liệu giả
    data = torch.randn(128, 1000).to(device)

    # Chạy vài vòng lặp tính toán
    start_time = time.time()
    for _ in range(1000):
        output = model(data)
        loss = output.sum() # Phép toán đơn giản
        loss.backward() # Bỏ qua backward để đơn giản hóa
    end_time = time.time()
    print(f"GPU test finished in {end_time - start_time:.4f} seconds.")

    # Theo dõi nvidia-smi khi script này chạy
else:
    print("Cannot run GPU test, CUDA not available.")