from argparse import ArgumentParser
from pathlib import Path
import sys

import torch
from torch import nn
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.models import get_model
from src.data import get_data_loaders

# Global variables
train_batch_idx = -1
max_val_acc = 0


def run_training(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    class_loss_fn: nn.Module,
    reg_loss_fn: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    ckpts_path: Path,
    device: str,
    loss_weight_class=1.0,
    loss_weight_reg=1.0
):
    """Main training loop"""
    for epoch_idx in tqdm(range(num_epochs), leave=True):
        model.train()
        training_epoch(
            model=model,
            class_loss_fn=class_loss_fn,
            reg_loss_fn=reg_loss_fn,
            optimizer=optimizer,
            epoch_idx=epoch_idx,
            device=device,
            dl_train=dl_train,
            loss_weight_class=loss_weight_class,
            loss_weight_reg=loss_weight_reg
        )

        model.eval()
        validation_epoch(
            model=model,
            epoch_idx=epoch_idx,
            device=device,
            dl_val=dl_val,
            ckpt_path=Path(ckpts_path),
            class_loss_fn=class_loss_fn,
            reg_loss_fn=reg_loss_fn,
            loss_weight_class=loss_weight_class,  
            loss_weight_reg=loss_weight_reg      
        )


def training_epoch(
    model: nn.Module,
    class_loss_fn: nn.Module,
    reg_loss_fn: nn.Module,
    optimizer: Optimizer,
    epoch_idx: int,
    device: torch.device,
    dl_train: DataLoader,
    loss_weight_class: float,   
    loss_weight_reg: float     
):
    global train_batch_idx
    for (train_batch_idx, batch) in enumerate(tqdm(dl_train, leave=False), start=train_batch_idx + 1):
        imgs, class_targets_int, class_targets, reg_targets = batch
        imgs, class_targets_int, reg_targets = imgs.to(device), class_targets_int.to(device), reg_targets.to(device)

        class_preds, reg_preds = model(imgs)
        loss_classification = class_loss_fn(class_preds, class_targets_int)
        loss_regression = reg_loss_fn(reg_preds, reg_targets)

        loss = (loss_classification * loss_weight_class + loss_regression * loss_weight_reg) / (loss_weight_class + loss_weight_reg)  # Combine both losses

        #optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.zero_grad()

        wandb.log({
            'Train/Loss_Classification': loss_classification.item(),
            'Train/Loss_Regression': loss_regression.item(),
            'Train/Loss': loss.item(),
            'epoch': epoch_idx,
            'batch_idx': train_batch_idx
        })


@torch.no_grad()
def validation_epoch(
    model: nn.Module,
    epoch_idx: int,
    device: torch.device,
    dl_val: DataLoader,
    ckpt_path: Path,
    class_loss_fn: nn.Module,  
    reg_loss_fn: nn.Module,  
    loss_weight_class: float,  
    loss_weight_reg: float     
):
    """Validation function that computes classification accuracy and regression loss separately"""
    accuracies_class = []
    accuracies_reg = []
    accuracies_reg_target = []
    class_losses = []
    reg_losses = []
    reg_mse_errors = []
    reg_mae_errors = []

    # Initialize the metric
    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)

    for batch_idx, batch in enumerate(tqdm(dl_val, leave=False)):
        imgs, class_targets_int, class_targets, reg_targets = batch
        imgs, class_targets_int, reg_targets = imgs.to(device), class_targets_int.to(device), reg_targets.to(device)

        # Forward pass
        class_preds, reg_preds = model(imgs)

        # Compute losses
        class_loss = class_loss_fn(class_preds, class_targets_int)
        reg_loss = reg_loss_fn(reg_preds, reg_targets)

        class_losses.append(class_loss.item())
        reg_losses.append(reg_loss.item())

        # Compute classification accuracy
        class_preds = class_preds.argmax(dim=1)
        acc_class = (class_preds == class_targets_int).float().mean()
        accuracies_class.append(acc_class.item())

        # Compute MSE (MAE) loss for regression
        reg_mse_error = mse(reg_preds, reg_targets).item() #Used
        reg_mae_error = mae(reg_preds, reg_targets).item() #Extra information

        reg_mse_errors.append(reg_mse_error)
        reg_mae_errors.append(reg_mae_error)

        accuracies_reg.append(reg_mse_error)
        accuracies_reg_target.append(mse(torch.zeros_like(reg_targets), reg_targets).item())

    # Compute mean metrics
    mean_acc_class = sum(accuracies_class) / len(accuracies_class)
    mean_acc_reg = 1 - (sum(accuracies_reg) / len(accuracies_reg)) / (sum(accuracies_reg_target) / len(accuracies_reg_target))
    mean_acc = ((loss_weight_class*mean_acc_class)+(loss_weight_reg*mean_acc_reg))/(loss_weight_class+loss_weight_reg)
    mean_class_loss = sum(class_losses) / len(class_losses)
    mean_reg_loss = sum(reg_losses) / len(reg_losses)
    mean_reg_mse_errors = sum(reg_mse_errors) / len(reg_mse_errors)
    mean_reg_mae_errors = sum(reg_mae_errors) / len(reg_mae_errors)

    # Log validation metrics separately
    wandb.log({
        "Val/Accuracy_Class": mean_acc_class,
        "Val/Accuracy_Reg": mean_acc_reg,
        "Val/Accuracy" : mean_acc,
        "Val/Classification_Loss": mean_class_loss,
        "Val/Regression_Loss": mean_reg_loss,
        "Val/MSE_error": mean_reg_mse_errors,
        "Val/MAE_error": mean_reg_mae_errors,
        "epoch": epoch_idx,
    })

    ckpt_path.mkdir(parents=True, exist_ok=True)
    suffix = '.pth'

    # Always save last model
    torch.save(model.state_dict(),
               ckpt_path / f'model_{wandb.run.id}_last{suffix}')
    
    # Save copy if accuracy increased
    global max_val_acc
    if mean_acc > max_val_acc:
        max_val_acc = mean_acc

        prefix = f'model_{wandb.run.id}_ep'

        # Remove previously created checkpoint(s)
        for p in ckpt_path.glob(f'{prefix}*{suffix}'):
            p.unlink()

        # Save checkpoint
        torch.save(model.state_dict(),
                   ckpt_path / f'{prefix}{epoch_idx}{suffix}')

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_name", default="resnet18", help="Model type")
    parser.add_argument("--model_weights", default="DEFAULT", help="Pretrained weights")

    # Paths
    parser.add_argument("--ckpts_path", default="./ckpts", help="Checkpoint directory")
    parser.add_argument("--load_ckpt", default=None, help="Load model checkpoint")

    # Dataset
    parser.add_argument("--data_path", default="../dataset", help="Dataset path")
    parser.add_argument("--csv_path", default="Wear_data.csv", help="Wear file")
    parser.add_argument("--num_folds", type=int, default=5, help="K-Folds for cross-validation")
    parser.add_argument("--val_fold", type=int, default=0, help="Validation fold index")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument("--split", type=float, default=0.9, help="Train-validation split")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="Training epochs")

    # Loss weighting
    parser.add_argument("--loss_weight_class", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--loss_weight_reg", type=float, default=1.0, help="Regression loss weight")

    # Logging
    parser.add_argument("--wandb_entity", default="YOUR_WANDB_USER_NAME", help="WandB entity")
    parser.add_argument("--wandb_project", default="YOUR_PROJECT_NAME", help="WandB project name")

    args = parser.parse_args()

    wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))
    print(wandb.run.id)

    # Load model
    model = get_model(name=args.model_name, weights=args.model_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load checkpoint
    if args.load_ckpt is not None:
        model.load_state_dict(torch.load(args.load_ckpt))

    # Optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Loss functions
    class_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
   
    
    # Get data loaders
    dl_train, dl_val, dl_test = get_data_loaders(
        data_path=args.data_path,
        csv_path=args.csv_path,
        size=args.size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        num_folds=args.num_folds,
        val_fold=args.val_fold,
        split = args.split,
    )

    # Run training
    run_training(
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        class_loss_fn=class_loss_fn,
        reg_loss_fn=reg_loss_fn,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        ckpts_path=args.ckpts_path,
        device=device,
        loss_weight_class=args.loss_weight_class,
        loss_weight_reg=args.loss_weight_reg,
    )
