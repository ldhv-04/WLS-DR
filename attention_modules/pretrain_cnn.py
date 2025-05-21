# pretrain_cnn.py (Cập nhật hoàn chỉnh)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Thư viện hiển thị progress bar
import argparse # Thêm thư viện argparse
import sys
import os # Thêm os để kiểm tra file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from configs import *   
import prepare_data.data_loader
import models.cnn_model as cnn_model
import configs.utils as utils

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.eval()
    loop = tqdm(loader, leave=True, desc="Training") # Thêm mô tả cho progress bar
    running_loss = 0.0
    processed_samples = 0
    batch_count = 0

    for batch_idx, batch_data in enumerate(loop):
        if batch_data is None or len(batch_data) != 2:
             print(f"Warning: Skipping invalid batch (index {batch_idx}) from DataLoader.")
             continue # Bỏ qua batch lỗi

        data, targets = batch_data
        if data is None or data.nelement() == 0:
             print(f"Warning: Skipping empty or None data in batch (index {batch_idx}).")
             continue # Bỏ qua batch rỗng

        data, targets = data.to(device), targets.to(device)

        # Forward
        scores = model(data)
        loss = loss_fn(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0) # Tính loss tổng theo số sample
        processed_samples += data.size(0)
        batch_count += 1

        # Cập nhật progress bar với loss trung bình của batch hiện tại
        loop.set_postfix(loss=loss.item())

    if batch_count == 0:
         print("Warning: No valid batches processed in training epoch.")
         return float('inf') # Trả về giá trị lỗi

    avg_loss = running_loss / processed_samples # Tính loss trung bình trên toàn bộ sample
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    running_loss = 0.0
    processed_samples = 0
    batch_count = 0
    # Có thể thêm tính toán Kappa score ở đây nếu cần đánh giá chi tiết hơn

    with torch.no_grad():
        loop = tqdm(loader, leave=True, desc="Validation") # Thêm mô tả
        for batch_idx, batch_data in enumerate(loop):
            if batch_data is None or len(batch_data) != 2:
                 print(f"Warning: Skipping invalid batch (index {batch_idx}) from DataLoader during validation.")
                 continue # Bỏ qua batch lỗi

            data, targets = batch_data
            if data is None or data.nelement() == 0:
                 print(f"Warning: Skipping empty or None data in batch (index {batch_idx}) during validation.")
                 continue # Bỏ qua batch rỗng

            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            loss = loss_fn(scores, targets)

            running_loss += loss.item() * data.size(0) # Tính loss tổng theo số sample
            processed_samples += data.size(0)
            batch_count += 1
            # Cập nhật progress bar với loss trung bình của batch hiện tại
            loop.set_postfix(val_loss=loss.item())

    if batch_count == 0:
         print("Warning: No valid batches processed in validation epoch.")
         return float('inf') # Trả về giá trị lỗi

    avg_loss = running_loss / processed_samples # Tính loss trung bình trên toàn bộ sample
    print(f"Average Validation Loss: {avg_loss:.4f}")
    return avg_loss


def main(): # Đưa logic chính vào hàm main
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description='Pretrain CNN for DR Grading')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')
    args = parser.parse_args()
    # -----------------------

    print("Device:", configs.DEVICE)
    print("Loading data...")
    # Sử dụng file CSV đã làm sạch nếu có
    label_file_to_use = configs.LABEL_FILE
    cleaned_csv_path = configs.LABEL_FILE.replace('.csv', '_cleaned.csv')
    if os.path.exists(cleaned_csv_path):
        print(f"Using cleaned label file: {cleaned_csv_path}")
        label_file_to_use = cleaned_csv_path
    else:
        print(f"Using original label file: {configs.LABEL_FILE}")
        print("Consider running clean_csv.py if you encounter data loading issues.")

    train_loader, valid_loader = prepare_data.data_loader.get_dataloaders(
        label_file_to_use, # Sử dụng file label đã xác định
        configs.TRAIN_IMG_DIR,
        configs.BATCH_SIZE,
        configs.VALID_SPLIT,
        configs.IMG_SIZE,
        configs.NUM_WORKERS
    )

    print("Initializing model...")
    model = cnn_model.get_cnn_model().to(configs.DEVICE)

    # Loss và Optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)

    # --- Load Checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, best_val_loss = utils.load_checkpoint(args.resume, model, optimizer)
            print(f"Resuming training from epoch {start_epoch + 1}")
        else:
            print(f"Warning: Checkpoint file not found at '{args.resume}'. Starting from scratch.")
    else:
        print("Starting training from scratch (or default weights).")
    # -----------------------

    print("Starting pre-training...")
    # Bắt đầu vòng lặp từ start_epoch (là epoch cuối cùng đã hoàn thành, nên bắt đầu từ epoch + 1)
    for epoch in range(start_epoch, configs.PRETRAIN_EPOCHS):
        current_epoch_display = epoch + 1 # Epoch hiển thị cho người dùng (bắt đầu từ 1)
        print(f"\n--- Epoch {current_epoch_display}/{configs.PRETRAIN_EPOCHS} ---")

        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, configs.DEVICE)
        val_loss = validate_one_epoch(valid_loader, model, loss_fn, configs.DEVICE)

        # Lưu model tốt nhất dựa trên validation loss
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': current_epoch_display, # Lưu epoch hiện tại (đã hoàn thành)
                'best_val_loss': best_val_loss
            }
            # Luôn lưu checkpoint mới nhất với tên cố định để dễ resume
            utils.save_checkpoint(checkpoint, filename=configs.MODEL_SAVE_PATH.replace('.pth', '_latest.pth.tar'))
            # Có thể lưu thêm checkpoint theo epoch hoặc best loss nếu muốn
            utils.save_checkpoint(checkpoint, filename=configs.MODEL_SAVE_PATH.replace('.pth', f'_best.pth.tar'))
        else:
            print(f"Validation loss did not improve from {best_val_loss:.4f}")
            # Vẫn có thể lưu checkpoint mới nhất để resume từ điểm dừng gần nhất
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': current_epoch_display,
                'best_val_loss': best_val_loss # Vẫn là best_val_loss cũ
            }
            utils.save_checkpoint(checkpoint, filename=configs.MODEL_SAVE_PATH.replace('.pth', '_latest.pth.tar'))


    print("Pre-training finished.")

if __name__ == "__main__":
    main() # Gọi hàm main