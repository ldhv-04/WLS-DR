# clean_csv.py (Ví dụ)
import pandas as pd
import os
from PIL import Image
import configs.configs as configs # Giả sử config.py chứa các đường dẫn

print(f"Reading CSV: {configs.LABEL_FILE}")
df = pd.read_csv(configs.LABEL_FILE)
original_count = len(df)
print(f"Original number of entries: {original_count}")

rows_to_drop = []
img_dir = configs.TRAIN_IMG_DIR

for index, row in df.iterrows():
    img_name = row['image'] + '.jpeg'
    img_path = os.path.join(img_dir, img_name)

    # Kiểm tra file tồn tại
    if not os.path.exists(img_path):
        # print(f"Missing: {img_path}")
        rows_to_drop.append(index)
        continue # Chuyển sang hàng tiếp theo

    # Kiểm tra file có bị hỏng không (thử mở)
    try:
        with Image.open(img_path) as img:
            img.verify() # Kiểm tra cấu trúc file cơ bản
        # Hoặc bạn có thể thử load hoàn toàn nếu verify chưa đủ
        # with Image.open(img_path) as img:
        #     img.load()
    except Exception as e: # Bắt lỗi chung khi mở/verify/load
        print(f"Corrupted or unreadable file {img_path}: {e}")
        rows_to_drop.append(index)

    if index % 1000 == 0: # In tiến trình
         print(f"Checked {index}/{original_count} entries...")


print(f"\nFound {len(rows_to_drop)} missing or corrupted files.")

if rows_to_drop:
    df_cleaned = df.drop(rows_to_drop)
    cleaned_count = len(df_cleaned)
    print(f"Number of entries after cleaning: {cleaned_count}")
    # Lưu file CSV đã làm sạch
    cleaned_csv_path = configs.LABEL_FILE.replace('.csv', '_cleaned.csv')
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned CSV saved to: {cleaned_csv_path}")
    print(f"\nIMPORTANT: Update LABEL_FILE in config.py to '{cleaned_csv_path}' before running training again.")
else:
    print("No missing or corrupted files found based on checks.")

# Chạy script này một lần: python clean_csv.py
# Sau đó cập nhật config.LABEL_FILE thành đường dẫn file CSV mới