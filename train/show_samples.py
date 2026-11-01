import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load the list of training images
df = pd.read_csv("data/split/train.csv")
print(df.columns)

# 2. Get the 6 class names, sorted
classes = sorted(df["class"].unique())
print("classes:", classes)

# 3. Make an empty grid: 6 rows x 5 columns
fig, axes = plt.subplots(6, 5, figsize=(10, 12))

# 4. Fill the grid
for r, name in enumerate(classes):              # one row per class
    rows = df[df["class"] == name].head(5)      # first 5 images of this class
    for c, path in enumerate(rows["path"]):     # one column per image
        img = Image.open(path)
        axes[r, c].imshow(img, cmap="gray")
        axes[r, c].axis("off")
        if c == 0:
            axes[r, c].set_title(name, loc="left")

# 5. Save the picture
Path("docs").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("docs/samples.png")
print("saved docs/samples.png")