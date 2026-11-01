# Load the dataset metadata, image reader, PyTorch dataset base class, and transforms.
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DefectDataset(Dataset):
    # Read image paths and labels, then choose transforms for training or evaluation.
    def __init__(self, csv_file, train):
        self.df=pd.read_csv(csv_file)
        
        if train:
            # Training transforms add augmentation to improve model generalization.
            self.transform=T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(10),
                                     T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            # Evaluation uses deterministic transforms so results are consistent.
            self.transform= T.Compose([T.Resize((224,224)), T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        # Return the number of images available in the CSV file.
        return len(self.df)
    
    def __getitem__(self, i):
        # Load one image, apply its transform, and return it with its numeric label.
        row = self.df.iloc[i]
        img_path = row["path"]
        label = row["label"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

if __name__=="__main__":
    # Run a small smoke test when this file is executed directly.
    ds=DefectDataset("data/split/train.csv", train=True)
    print(ds.df.head(5))
    img, label=ds[0]
    print(len(ds), img.shape, label)