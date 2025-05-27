from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Ignore 'DataFrame.swapaxes' is deprecated warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore palette images with transparency expressed in bytes warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class InsertDataset(Dataset):
    def __init__(self, data_path="../../dataset", split=0.9, subset = 'train', k=5, val_fold=0, transform=None, csv_file="Wear_data.csv"):
        data_path = Path(data_path)
        df_wear = pd.read_csv(csv_file)
        wear_dict = df_wear.set_index("Insert_Name")[["TOP", "LEFT", "RIGHT", "BOTTOM"]].to_dict(orient="index")

        # Create dataframe from data
        data = []
        for img_path in data_path.glob('*/*'):
            if img_path.stem in wear_dict:
                data.append({
                    'image': str(img_path),
                    'label': img_path.parent.name,
                    'Top': wear_dict[img_path.stem]["TOP"], 
                    'Left': wear_dict[img_path.stem]["LEFT"],  
                    'Right': wear_dict[img_path.stem]["RIGHT"],  
                    'Bottom': wear_dict[img_path.stem]["BOTTOM"] 
                })
            else:
                print(f"Skipping {img_path}: data not found")

        df = pd.DataFrame(data)

        # Create mapping from label to integer
        label_to_int = {
            label: i
            for i, label in enumerate(df['label'].unique())
        }
        
        # Split into train, test, val
        df_trainval, df_test = train_test_split(df, train_size=split, random_state=1)
        folds = np.array_split(df_trainval, k)
        df_val = folds[val_fold]
        train_folds = [fold for i, fold in enumerate(folds)if i != val_fold]
        df_train = pd.concat(train_folds)

        # Store attributes
        self.data_path = data_path
        self.label_to_int = label_to_int
        self.transform = transform
        self.subset = subset


        if subset == 'train':
            self.df = df_train.reset_index()
        elif subset == 'val':
            self.df = df_val.reset_index()
        elif subset == 'test':
            self.df = df_test.reset_index()
        else:
            raise ValueError(f'Unknown subset "{subset}"')
   
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['image'][idx]
        label = self.df['label'][idx]
        int_label = self.label_to_int[label]

        regression_target = np.array([
            self.df['Top'][idx],
            self.df['Left'][idx],
            self.df['Right'][idx],
            self.df['Bottom'][idx]
        ], dtype=np.float32)

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, int_label, label, regression_target

