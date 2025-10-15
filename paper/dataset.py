import json
import os.path

import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ViTImageProcessor, AutoImageProcessor


class Flick8kDataset(Dataset):
    def __init__(self, img_dir, samples, tokenizer=None, transform=None, augmentation=None):
        super().__init__()
        self.img_dir = img_dir
        self.samples = samples
        self.tokenizer = tokenizer
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img_name, caption, img_id = self.samples[item]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        if self.augmentation:
            img = self.augmentation(img)
        features = self.transform(images=img, return_tensors='pt').pixel_values.squeeze(0)

        labels = torch.tensor(self.tokenizer.numericalize(caption))
        labels = labels.squeeze(0)
        return {
            'pixel_values': features,
            'labels' : labels,
            'img_id' : img_id
        }

def build_paper_dataset(img_dir, tokenizer, json_path, split='train', ):
    # build for the train & val & test
    with open(json_path, "r") as f:
        data = json.load(f)
    samples = []
    for item in data['images']:
        if item['split'] == split:
            if split == 'test':
                samples.append((item['filename'], '', item['imgid']))
            else:
                for sentence in item['sentences']:
                    samples.append((item['filename'], sentence['raw'], item['imgid']))

    # data augmentation & fit for model
    if split == 'train':
        augmentations = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    else :
        augmentations = None
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    dataset = Flick8kDataset(img_dir=img_dir, samples=samples, tokenizer=tokenizer, transform=processor, augmentation=augmentations )

    return dataset
