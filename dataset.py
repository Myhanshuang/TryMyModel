import json
import os.path
from pyexpat import features

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import KFold
from transformers import ViTImageProcessor


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
        if isinstance(self.transform, ViTImageProcessor):
            features = self.transform(images=img, return_tensors='pt').pixel_values.squeeze(0)
        else :
            features = self.transform(img)

        labels = self.tokenizer(
            caption,
            padding = "max_length",
            max_length = 50,
            truncation=True,
            return_tensors="pt"#why do you carry the batch dim????
        ).input_ids
        labels = labels.squeeze(0)
        return {
            'pixel_values': features,
            'labels' : labels,
            'img_id' : img_id
        }

def build_dataset(img_dir, tokenizer, json_path, split='train', ):
    # build for the train & val & test
    with open(json_path, "r") as f:
        data = json.load(f)
    samples = []
    for item in data['images']:
        if item['split'] == split:
            for sentence in item['sentences']:
                samples.append((item['filename'], sentence['raw'], item['imgid']))

    # data augmentation & fit for model
    if split == 'train':
        transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)), # 随机裁剪
                transforms.RandomHorizontalFlip(), # 随机水平翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 颜色抖动
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # ViT通常使用0.5的均值和标准差
            ])
    else :
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = Flick8kDataset(img_dir=img_dir, samples=samples, tokenizer=tokenizer, transform=transform, augmentation=None)

    return dataset

def build_dataset_pretrain(img_dir, tokenizer, json_path, split='train'):
    with open(json_path, "r") as f:
        data = json.load(f)
    samples = []
    for item in data['images']:
        if item['split'] == split:
            for sentence in item['sentences']:
                samples.append((item['filename'], sentence['raw'], item['imgid']))

    # 统一加载与模型配套的 image_processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    augmentations = None
    # 只为训练集定义数据增强
    if split == 'train':
        augmentations = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    dataset = Flick8kDataset(
        img_dir=img_dir,
        samples=samples,
        tokenizer=tokenizer,
        transform=image_processor,  # 总是传入
        augmentation=augmentations  # 只在训练时传入
    )

    return dataset

def build_k_fold_dataset(img_dir, tokenizer, json_path, n_splits, seed):
    # return a list of the pair (train, val) dataset
    with open(json_path, "r") as f:
        data = json.load(f)
    samples = []
    for item in data['images']:
        if item['split'] in ['train', 'val']:
            for sentence in item['sentences']:
                samples.append((item['filename'], sentence['raw'], item['imgid']))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_transform = transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ViT通常使用0.5的均值和标准差
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    all_dataset = []

    for train_idx, val_idx in kf.split(samples):
        train_samples = [samples[idx] for idx in train_idx]
        val_samples = [samples[idx] for idx in val_idx]

        train_dataset = Flick8kDataset(img_dir=img_dir, samples=train_samples, tokenizer=tokenizer, transform=train_transform)
        val_dataset = Flick8kDataset(img_dir=img_dir, samples=val_samples, tokenizer=tokenizer, transform=val_transform)

        all_dataset.append((train_dataset, val_dataset))

    return all_dataset