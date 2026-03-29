import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# หา root ของโปรเจกต์จากตำแหน่งไฟล์นี้
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE_DIR, "Data", "cifar-10Process", "cifar-10")


def find_image_file(folder, image_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(folder, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def check_data():
    csv_path = os.path.join(ROOT, "trainLabels.csv")
    train_dir = os.path.join(ROOT, "train", "train")
    test_dir = os.path.join(ROOT, "test", "test")

    print("Checking dataset...")
    print("ROOT :", ROOT)
    print("CSV  :", csv_path)
    print("Train:", train_dir)
    print("Test :", test_dir)

    if not os.path.exists(csv_path):
        print("❌ trainLabels.csv not found")
        return False

    if not os.path.isdir(train_dir):
        print("❌ train/train folder not found")
        return False

    if not os.path.isdir(test_dir):
        print("❌ test/test folder not found")
        return False

    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print("Train file count:", len(train_files))
    print("Test file count :", len(test_files))

    if len(train_files) > 0:
        print("Sample train files:", train_files[:5])
    if len(test_files) > 0:
        print("Sample test files:", test_files[:5])

    df = pd.read_csv(csv_path)
    sample_id = str(df.iloc[0]["id"])
    sample_path = find_image_file(train_dir, sample_id)

    if sample_path is None:
        print(f"❌ could not find image for id {sample_id}")
        return False

    print("✅ sample image found:", sample_path)
    return True


class TrainDataset(Dataset):
    def __init__(self, transform=None):
        self.csv_path = os.path.join(ROOT, "trainLabels.csv")
        self.train_dir = os.path.join(ROOT, "train", "train")
        self.transform = transform

        self.df = pd.read_csv(self.csv_path)

        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["id"])
        label_name = row["label"]
        label_idx = self.class_to_idx[label_name]

        image_path = find_image_file(self.train_dir, image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for id {image_id} in {self.train_dir}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.test_dir = os.path.join(ROOT, "test", "test")
        self.transform = transform

        self.image_files = sorted(
            [
                f for f in os.listdir(self.test_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
            key=lambda x: int(os.path.splitext(x)[0])
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.test_dir, file_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image_id = os.path.splitext(file_name)[0]
        return image, image_id


def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    if not check_data():
        raise SystemExit("Fix dataset first.")

    transform = get_transform()

    train_dataset = TrainDataset(transform=transform)
    test_dataset = TestDataset(transform=transform)

    print("Train size:", len(train_dataset))
    print("Classes:", train_dataset.classes)
    print("Test size:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(train_loader))

    print("Train batch image shape:", images.shape)
    print("Train batch labels:", labels)