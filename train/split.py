# Import Path for building filesystem paths in a platform-independent way.
from pathlib import Path

# Use pandas for tabular data processing during dataset preparation.
import pandas as pd

# Use a reproducible helper to divide the dataset into training and test sets.
from sklearn.model_selection import train_test_split

# Location of the original NEU-DET dataset.
RAW_DIR = Path("data/raw")

# Destination for the generated dataset split.
OUT_DIR = Path("data/split")

# Keep dataset splitting consistent each time the script is run.
SEED = 42


# Find every JPG image below the raw dataset directory and record its class.
def collect_images(raw_dir):
    # Store the image file paths and their corresponding class names separately.
    paths=[]
    classs_name=[]

    # Search through all nested folders for JPG files.
    for i in  raw_dir.rglob("*.jpg"):
        # Save the image path as text for later processing.
        paths.append(str(i))

        # The image's parent folder identifies its defect class.
        classs_name.append(i.parent.name)
    return paths,classs_name
    
def main():
    # Collect all image paths and the defect class associated with each image.
    paths, classs_name=collect_images(RAW_DIR)
    print("image found ", len(paths))

    # Create a stable numeric label for every defect class.
    classes = sorted(set(classs_name))
    #print("classes " , classes)
    class_to_idx={name: i for i,name in enumerate(classes)}
    #print(class_to_idx)
    labels=[class_to_idx[name] for name in classs_name]
    #print(labels)

    # Reserve 70% for training and 30% for validation and testing.
    train_paths, rest_paths, train_labels, rest_labels, train_classes, rest_classes = train_test_split(
        paths, labels, classs_name, test_size=0.3, stratify=labels, random_state=SEED)

    # Divide the remaining 30% equally into validation and test sets.
    val_paths, test_paths, val_labels, test_labels, val_classes, test_classes = train_test_split(
        rest_paths, rest_labels, rest_classes, test_size=0.5, stratify=rest_labels, random_state=SEED)

    print(len(train_paths), len(val_paths), len(test_paths))

    # Create the output directory if it does not already exist.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep each split's paths, labels, and class names together for export.
    splits=[("train", train_paths, train_labels, train_classes),
            ("val", val_paths, val_labels, val_classes),
            ("test", test_paths, test_labels, test_classes)]

    # Save each dataset split as a CSV file and print its class distribution.
    for split_name, p,t,l in splits:
        df=pd.DataFrame({"path":p, "label":t, "class":l})
        df.to_csv(OUT_DIR / f"{split_name}.csv", index=False)
        print(split_name, len(df))
        print(df["class"].value_counts())
       
        
    
    

    
if __name__ == "__main__":
    main()

    