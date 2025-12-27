import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class LowLightDataset(data.Dataset):
    """低光照图像数据集加载器"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 转换为tensor (0-1范围)
        image = transforms.ToTensor()(image)
        return image

def get_dataloader(root_dir, batch_size=8, num_workers=4):
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = LowLightDataset(root_dir, transform=transform)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader