from typing import List
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from .insert_dataset import InsertDataset
import torchvision.transforms.functional as F


def get_data_loaders(
    data_path: str,
    csv_path: str,
    size: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    num_folds: int,
    val_fold: int,
    split: float,
    norm_mean: List[int] = [0.485, 0.456, 0.406],
    norm_std: List[int] = [0.229, 0.224, 0.225],
):
    
    # Create transforms (Images are 640x960)
    transforms = v2.Compose([
    v2.ToImage(), 
    v2.Lambda(lambda img: F.affine(img, angle=0, translate=(-35, 0), scale=1, shear=0)), 
    v2.Resize(size, antialias=True), 
    v2.CenterCrop(size),
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize(mean=norm_mean, std=norm_std), 
])

    # Create datasets
    ds_train = InsertDataset(data_path=data_path, split = split, subset ='train', k=num_folds, val_fold=val_fold, transform=transforms, csv_file=csv_path)
    ds_val = InsertDataset(data_path=data_path, split = split, subset ='val', k=num_folds, val_fold=val_fold, transform=transforms, csv_file=csv_path)
    ds_test = InsertDataset(data_path=data_path, split = split, subset ='test', k=num_folds, val_fold=val_fold, transform=transforms, csv_file=csv_path)
    
    # Create data loaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return dl_train, dl_val, dl_test
