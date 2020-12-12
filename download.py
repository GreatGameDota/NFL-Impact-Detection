import os

# os.system("kaggle datasets download -d greatgamedota/torchvision-modified")
# os.system("unzip -q torchvision-modified.zip")
# os.system("rm torchvision-modified.zip")

os.system("kaggle datasets download -d greatgamedota/nfl-impact-detection-train-images")
os.system("unzip -q nfl-impact-detection-train-images.zip -d 'data/'")
os.system("rm nfl-impact-detection-train-images.zip")

os.system("kaggle datasets download -d greatgamedota/nfl-impact-detection-train-csv")
os.system("unzip -q nfl-impact-detection-train-csv.zip -d 'data/'")
os.system("rm nfl-impact-detection-train-csv.zip")