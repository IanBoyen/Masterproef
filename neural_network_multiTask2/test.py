from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.models import get_model
from src.data import get_data_loaders
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from tabulate import tabulate


def run_test(model: torch.nn.Module,
             dl_test: DataLoader,
             device: str,
             classification_threshold: float = 0.5,  # For classification thresholding
             reg_accuracy_absolute_range: float = 50,  # Tolerance for regression accuracy
             reg_accuracy_relative_percentage_range: float = 0.2,  # Tolerance percentage for regression accuracy
             reg_binary_threshold: int = 200): # Binary Tolerance for regression accuracy
    model.eval()

    # Initialize lists for storing true labels and predictions
    gt_class = []
    preds_class = []
    gt_reg = []
    preds_reg = []

    # Initialize counters for classification accuracy and regression accuracy
    correct_class = 0
    total_class = 0

    # Create dictionaries for each variable to store values for each insert type
    correct_range_reg = {}
    correct_binary_reg_dict = {}
    falseNegative_binary_reg_dict = {}
    falsePositive_binary_reg_dict = {}
    total_reg_dict = {}
    classes_dict = {}
    mse_dict = {}
    mae_dict = {}

    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)

    # Dictionary to store regression metrics by insert type
    regression_results = {}

    with torch.no_grad():
        for images, class_labels_int, class_labels, reg_labels in tqdm(dl_test):  # Assuming the dataloader returns both class and reg labels
            images, class_labels_int, reg_labels = images.to(device), class_labels_int.to(device), reg_labels.to(device)

            #Initialing dictionaries
            for i in range(len(class_labels)):
                if class_labels_int[i].item() not in classes_dict:
                    classes_dict[class_labels_int[i].item()] = class_labels[i]
                    correct_range_reg[class_labels_int[i].item()] = 0
                    correct_binary_reg_dict[class_labels_int[i].item()] = 0
                    falseNegative_binary_reg_dict[class_labels_int[i].item()] = 0
                    falsePositive_binary_reg_dict[class_labels_int[i].item()] = 0
                    total_reg_dict[class_labels_int[i].item()] = 0
                    mse_dict[class_labels_int[i].item()] = 0
                    mae_dict[class_labels_int[i].item()] = 0

            # Forward pass
            output = model(images)
            class_pred = output[0]  # Assuming first output is for classification
            reg_pred = output[1]  # Assuming second output is for regression

            # Classification: Predicted class
            _, predicted_class = class_pred.max(1)  # Get predicted class
            correct_class += (predicted_class == class_labels_int).sum().item()
            total_class += class_labels_int.size(0)

            # Regression: Loop through each regression target [Top, Left, Right, Bottom] for each image
            for true_value, predicted_value, insert_type_int in zip(reg_labels.cpu().numpy(), reg_pred.cpu().numpy(), class_labels_int.cpu().numpy()):
                # Perform regression accuracy
                abs_diff= abs(true_value-predicted_value)
                if ((abs_diff / (true_value + 1e-6) <= reg_accuracy_relative_percentage_range) or (abs_diff <= reg_accuracy_absolute_range)):
                    correct_range_reg[insert_type_int] += 1

                # Perform binary threshold evaluation
                if true_value >= reg_binary_threshold:
                    if predicted_value >= reg_binary_threshold:
                        correct_binary_reg_dict[insert_type_int] += 1
                    else:
                        falseNegative_binary_reg_dict[insert_type_int] += 1
                else:
                    if predicted_value < reg_binary_threshold:
                        correct_binary_reg_dict[insert_type_int] += 1
                    else:
                        falsePositive_binary_reg_dict[insert_type_int] += 1

                #total targets
                total_reg_dict[insert_type_int] += 1

                #mse and mae
                mse_dict[insert_type_int] = mse(torch.tensor([predicted_value], device=device), torch.tensor([true_value], device=device))
                mae_dict[insert_type_int] = mae(torch.tensor([predicted_value], device=device), torch.tensor([true_value], device=device))

            # Collect true labels and predictions for classification report
            gt_class.extend(class_labels_int.cpu().numpy())
            preds_class.extend(predicted_class.cpu().numpy())
            gt_reg.extend(reg_labels.cpu().numpy())
            preds_reg.extend(reg_pred.cpu().numpy())

            # Store metrics for each insert type (per batch)
            for key in classes_dict:
                insert_type = classes_dict[key]
                if insert_type not in regression_results:
                    regression_results[insert_type] = {
                        "correct_range": 0,
                        "correct_binary": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "mse": 0,
                        "mae": 0,
                        "total": 0
                    }

                # Aggregate the values into regression_results for that insert type
                regression_results[insert_type]["correct_range"] += correct_range_reg[key]
                regression_results[insert_type]["correct_binary"] += correct_binary_reg_dict[key]
                regression_results[insert_type]["false_positive"] += falsePositive_binary_reg_dict[key]
                regression_results[insert_type]["false_negative"] += falseNegative_binary_reg_dict[key]
                regression_results[insert_type]["mse"] += mse_dict[key]
                regression_results[insert_type]["mae"] += mae_dict[key]
                regression_results[insert_type]["total"] += total_reg_dict[key]


    #Overall Report
    print("Overall Report:")
    print("-" * 50)

    # Classification accuracy
    classification_accuracy = 100 * correct_class / total_class
    print(f"Classification Accuracy: {classification_accuracy:.2f}%")

    # Regression accuracy (based on tolerance threshold)
    regression_accuracy = 100 * sum(i["correct_range"] for i in regression_results.values()) / sum(i["total"] for i in regression_results.values())
    print(f"Regression Accuracy Range (within {reg_accuracy_relative_percentage_range*100} percentage or {reg_accuracy_absolute_range}µm tolerance): {regression_accuracy:.2f}%")

    # Regression correctness (binary threshold)
    binary_reg_accuracy = 100 * sum(i["correct_binary"] for i in regression_results.values()) / sum(i["total"] for i in regression_results.values())
    print(f"Binary Regression Accuracy: (Threshold = {reg_binary_threshold}µm): {binary_reg_accuracy:.2f}%")
    print(f"False Positive Rate (Not worn-out but detected): {100 * sum(i['false_positive'] for i in regression_results.values()) / sum(i['total'] for i in regression_results.values()):.2f}%")
    print(f"False Negative Rate (Worn-out but not detected): {100 * sum(i['false_negative'] for i in regression_results.values()) / sum(i['total'] for i in regression_results.values()):.2f}%") #worst of the 2 false detections

    # MSE and MAE for regression
    print(f"Mean Squared Error (MSE): {sum(i['mse'] for i in regression_results.values()) / sum(i['total'] for i in regression_results.values()):.4f}")
    print(f"Mean Absolute Error (MAE): {sum(i['mae'] for i in regression_results.values()) / sum(i['total'] for i in regression_results.values()):.4f}")

    #Total
    print(f"Total Targets: {sum(i['total'] for i in regression_results.values())}")


    #Detailed Report
    print("\nDetailed Report:")
    print("-" * 50)

    # Print detailed classification report
    classes = list(dl_test.dataset.label_to_int.keys())
    report_class = classification_report(gt_class, preds_class, target_names=classes)
    print("Classification Report:\n", report_class)


    # Print detailed regression report
    regression_table = []
    headers = ["Insert Type", "Correct Range", "Binary Acc", "False Positive", "False Negative", "MSE", "MAE", "Total"]

    for insert_type, stats in regression_results.items():
        row = [
            insert_type,
            f"{(stats['correct_range'] / stats['total']) * 100:.2f}%",
            f"{(stats['correct_binary'] / stats['total']) * 100:.2f}%",
            f"{(stats['false_positive'] / stats['total']) * 100:.2f}%",
            f"{(stats['false_negative'] / stats['total']) * 100:.2f}%",
            f"{stats['mse'] / stats['total']:.2f}",
            f"{stats['mae'] / stats['total']:.2f}",
            stats['total']
        ]
        regression_table.append(row)

    print("\nRegression Report Per Insert Type:")
    print(tabulate(regression_table, headers=headers, tablefmt="pretty"))

    return classification_accuracy, regression_accuracy

if __name__ == '__main__':
    parser = ArgumentParser()

    # Model
    parser.add_argument(
        '--model_name',
        help='The name of the model to use for the classifier.',
        default='resnet18',
    )
    parser.add_argument(
        '--model_weights',
        help='The pretrained weights to load. If None, the weights are '
        'randomly initialized. See also '
        'https://pytorch.org/vision/stable/models.html.',
        default='DEFAULT'
    )

    # Checkpoints
    parser.add_argument(
        '--ckpts_path',
        default='./ckpts',
        help='The directory to save checkpoints.' 
    )
    parser.add_argument(
        '--load_ckpt',
        default=None,
        help='The path to load model checkpoint weights from.' 
    )

    # Data path
    parser.add_argument(
        '--data_path',
        default='../dataset',
        help='Path to the dataset',
    )
    parser.add_argument(
        '--csv_path',
        default='Wear_data.csv',
        help='Wear file',
    )

    # Split data
    parser.add_argument(
        '--split',
        default=0.9,
        help='The split between training_val and test data.',
        type=float
    )

    # K-Fold args
    parser.add_argument(
        '--num_folds',
        default=5,
        help='The number of folds to use for cross-validation.',
        type=int
    )
    parser.add_argument(
        '--val_fold',
        default=0,
        help='The index of the validation fold. '
        'If None, all folds are used for training.',
        type=int
    )

    # Data loader args
    parser.add_argument(
        '--batch_size',
        default=32,
        help='The training batch size.',
        type=int
    )
    parser.add_argument(
        '--val_batch_size',
        default=32,
        help='The validation batch size.',
        type=int
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    args = parser.parse_args()

    model = get_model(
        name=args.model_name,
        weights=args.model_weights,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and dataset
    model = model.to(device)

    if args.load_ckpt is not None:
        model.load_state_dict(torch.load(args.load_ckpt))

    # Get the data loaders
    dl_train, dl_val, dl_test = get_data_loaders(
        data_path=args.data_path,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        num_folds=args.num_folds,
        val_fold=args.val_fold,
        split=args.split,
    )

    # Run testing
    run_test(
        model=model,
        dl_test=dl_test,
        device=device
    )
