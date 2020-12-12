import os

os.system("kaggle datasets download -d greatgamedota/lyft-scenes")
os.system("unzip -q lyft-scenes.zip -d 'data/'")
os.system("rm lyft-scenes.zip")
