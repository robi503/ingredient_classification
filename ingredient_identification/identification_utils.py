import os
from torchvision.io import read_image
import torch
import json
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors

def load_images(base_dir):
    """
    Load images from the specified directory.

    Args:
        base_dir (str): The base directory containing the images.

    Returns:
        list: A list of loaded images.
    """
    images = []
    for file_name in os.listdir(base_dir):
        images.append(file_name)
    print(f'Images length {len(images)}')
    return images


def create_data_tuples(base_dir, annotations_file):
    """
    Load annotations for the given images from a COCO format JSON file.

    Args:
        images (list): List of images.
        annotations_file (str): Path to the COCO format JSON file containing annotations.

    Returns:
        list: List of targets (annotations) corresponding to each image.
    """
    data_tuples = []

    images = load_images(base_dir)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    image_info = data['images']
    for file_name in images:
        # Find annotations for the current image
        for info in image_info:
            if info['file_name'] == file_name:
                image_id = info['id']
        image_annotations = [annotation for annotation in data["annotations"] if annotation["image_id"] == str(image_id)]
        
        # Initialize a list to store annotations for the current image
        annotations = []
        
        for annotation in image_annotations:
            bbox = annotation['bbox']
            x0, y0 = bbox[0], bbox[1]
            x1, y1 = x0 + bbox[2], y0 + bbox[3]
            annotation_dict = {
                'bbox': torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'image_id': torch.tensor(image_id, dtype=torch.int64),
                'area': torch.tensor(annotation['area'], dtype=torch.float32),
                'iscrowd': torch.tensor(annotation['iscrowd'], dtype=torch.int64),
            }
            annotations.append(annotation_dict)

        target = {
            'boxes': torch.stack([ann['bbox'] for ann in annotations], dim=0),
            'labels': torch.tensor([1] * len(annotations), dtype=torch.int64),
            'image_id': torch.tensor(image_id, dtype=torch.int64),
            'area': torch.tensor([ann['area'] for ann in annotations], dtype=torch.float32),
            'iscrowd': torch.tensor([ann['iscrowd'] for ann in annotations], dtype=torch.int64),
        }

        image_path = os.path.join(base_dir, file_name)
        image = read_image(image_path)
        data_tuples.append((image, target))
    print(f'Tuples length {len(data_tuples)}')
    return data_tuples


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, annotations_file, transforms=None):
        self.data_tuples = create_data_tuples(base_dir, annotations_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        # Load image and target from data_tuples
        image, target_dict = self.data_tuples[idx]

        # Extract target information
        boxes = target_dict['boxes']
        labels = target_dict['labels']
        image_id = target_dict['image_id']
        area = target_dict['area']
        iscrowd = target_dict['iscrowd']

        # Create image tensor object
        img = tv_tensors.Image(image)

        # Create target dictionary
        target = {
            'boxes': tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        # Apply transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data_tuples)
