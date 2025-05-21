# data_loader.py
import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import configs.configs as configs # Import cấu hình

# --- DI CHUYỂN collate_fn RA NGOÀI ---
# Cần collate_fn để xử lý trường hợp __getitem__ trả về None
def collate_fn(batch):
    # Lọc ra các sample bị lỗi (ví dụ: file not found trả về None)
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        # Trả về tensor rỗng nếu toàn bộ batch bị lỗi
        # Cần xác định shape phù hợp hoặc xử lý đặc biệt ở nơi gọi
        # Ví dụ đơn giản: trả về tuple tensor rỗng
        return torch.Tensor(), torch.Tensor()
    # Dùng default_collate cho phần còn lại
    return torch.utils.data.dataloader.default_collate(batch)
# --- KẾT THÚC PHẦN DI CHUYỂN ---

class EyePACSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
             # print(f"Warning: Index {idx} out of bounds for dataframe length {len(self.dataframe)}")
             return None # Trả về None nếu index không hợp lệ

        img_name_no_ext = self.dataframe.iloc[idx]['image']
        # Thử các phần mở rộng phổ biến nếu không chắc chắn
        possible_exts = ['.jpeg', '.png', '.jpg', '.tif', '.bmp']
        img_path = None
        for ext in possible_exts:
            potential_path = os.path.join(self.img_dir, img_name_no_ext + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            print(f"Warning: File not found for image ID {img_name_no_ext} in {self.img_dir}, returning None.")
            return None # Trả về None nếu không tìm thấy file

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"Warning: Error opening file {img_path}: {e}, returning None.")
             return None # Trả về None nếu lỗi đọc file


        # Nhãn là cấp độ DR (ví dụ: cột 'level')
        label = torch.tensor(float(self.dataframe.iloc[idx]['level']), dtype=torch.float32)
        # Vì dùng MSE loss, nhãn là float. Nếu dùng CrossEntropy, để int và dtype=torch.long

        if self.transform:
            image = self.transform(image)

        # Reshape label thành [1] nếu cần cho MSELoss
        label = label.unsqueeze(0)

        return image, label

def get_transforms(img_size):
    # Chuẩn hóa theo ImageNet mean và std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Transform cho tập huấn luyện (có augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # Thêm các augmentation khác nếu cần
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    # Transform cho tập validation/test (chỉ resize và chuẩn hóa)
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_test_transform

def get_dataloaders(label_file, img_dir, batch_size, valid_split, img_size, num_workers=4):
    try:
        df = pd.read_csv(label_file)
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_file}")
        return None, None

    # Kiểm tra xem cột 'image' và 'level' có tồn tại không
    if 'image' not in df.columns or 'level' not in df.columns:
        print(f"Error: Label file {label_file} must contain 'image' and 'level' columns.")
        return None, None

    # Loại bỏ các hàng có giá trị NaN trong cột 'image' hoặc 'level'
    df.dropna(subset=['image', 'level'], inplace=True)
    df = df.astype({'level': int}) # Đảm bảo level là int để stratify

    # Có thể cần tiền xử lý tên file trong df nếu nó không khớp (ví dụ: bỏ '_left'/'_right')
    # df['image'] = df['image'].str.replace('_left', '').str.replace('_right', '') # Ví dụ

    if len(df) == 0:
        print("Error: No valid data found in the label file after cleaning.")
        return None, None

    # Đảm bảo valid_split hợp lệ
    if not 0 < valid_split < 1:
         print("Error: valid_split must be between 0 and 1.")
         return None, None

    # Kiểm tra số lượng mẫu có đủ để phân tầng không
    if df['level'].nunique() > 1: # Chỉ stratify nếu có nhiều hơn 1 lớp
        min_samples_per_class = df['level'].value_counts().min()
        # `n_splits` cho StratifiedShuffleSplit hoặc `test_size` cho train_test_split cần ít nhất 2 mẫu/lớp
        if min_samples_per_class < 2:
             print("Warning: Some classes have less than 2 samples. Stratification might behave unexpectedly or fail. Consider not stratifying.")
             # Tùy chọn: không stratify nếu có lớp quá ít mẫu
             train_df, valid_df = train_test_split(
                 df,
                 test_size=valid_split,
                 random_state=42 # Để kết quả có thể tái lập
             )
        else:
             train_df, valid_df = train_test_split(
                 df,
                 test_size=valid_split,
                 random_state=42, # Để kết quả có thể tái lập
                 stratify=df['level'] # Phân tầng theo nhãn DR
             )
    else: # Nếu chỉ có 1 lớp, không cần stratify
        train_df, valid_df = train_test_split(
            df,
            test_size=valid_split,
            random_state=42
        )


    train_transform, val_test_transform = get_transforms(img_size)

    train_dataset = EyePACSDataset(train_df.reset_index(drop=True), img_dir, transform=train_transform) # Reset index sau khi split
    valid_dataset = EyePACSDataset(valid_df.reset_index(drop=True), img_dir, transform=val_test_transform) # Reset index

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn, # Sử dụng collate_fn đã định nghĩa ở top-level
        persistent_workers=True if num_workers > 0 else False # Giữ worker alive giữa các epoch
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False, # Không cần shuffle validation
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn, # Sử dụng collate_fn đã định nghĩa ở top-level
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    return train_loader, valid_loader




# Ví dụ cách sử dụng (chỉ chạy khi file này được thực thi trực tiếp)
if __name__ == "__main__":
    print(f"Running DataLoader test with num_workers={configs.NUM_WORKERS}")
    train_loader, valid_loader = get_dataloaders(
        configs.LABEL_FILE,
        configs.TRAIN_IMG_DIR,
        configs.BATCH_SIZE,
        configs.VALID_SPLIT,
        configs.IMG_SIZE,
        configs.NUM_WORKERS
    )

    if train_loader and valid_loader:
        print("\nTesting iteration over train_loader...")
        try:
            # Thử lặp qua một vài batch
            num_batches_to_test = 5
            count = 0
            for i, batch_data in enumerate(train_loader):
                if batch_data[0].nelement() == 0: # Kiểm tra batch rỗng từ collate_fn
                     print(f"Skipping empty batch {i+1}")
                     continue
                images, labels = batch_data
                print(f"Batch {i+1}: Image shape={images.shape}, Label shape={labels.shape}")
                count += 1
                if count >= num_batches_to_test:
                    break
            print(f"\nSuccessfully iterated over {count} train batches.")
        except Exception as e:
            print(f"\nAn error occurred during train_loader iteration: {e}")
            import traceback
            traceback.print_exc()

        print("\nTesting iteration over valid_loader...")
        try:
            # Thử lặp qua một vài batch
            num_batches_to_test = 5
            count = 0
            for i, batch_data in enumerate(valid_loader):
                 if batch_data[0].nelement() == 0: # Kiểm tra batch rỗng
                     print(f"Skipping empty batch {i+1}")
                     continue
                 images, labels = batch_data
                 print(f"Batch {i+1}: Image shape={images.shape}, Label shape={labels.shape}")
                 count += 1
                 if count >= num_batches_to_test:
                    break
            print(f"\nSuccessfully iterated over {count} validation batches.")
        except Exception as e:
            print(f"\nAn error occurred during valid_loader iteration: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to create DataLoaders.")