from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import os
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGBA')  # Ensure RGBA format
        image = image.convert('RGB')
        label = torch.tensor(self.dataframe.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
# Create DataFrames
def create_df(folder_path):
    all_images = []
    classes = [class_name for class_name in os.listdir(folder_path)]
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        all_images.extend([(os.path.join(class_path, file_name), classes.index(class_name)) for file_name in os.listdir(class_path)])
    df = pd.DataFrame(all_images, columns=['file_path', 'label'])
    return df

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])