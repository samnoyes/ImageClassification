from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import urllib
import sys

# Check if an argument is passed
if len(sys.argv) < 2:
    print("Error: No command-line argument passed.")
    sys.exit(1)

# Create model
model = timm.create_model('resnext101_32x8d', pretrained=True)
model.eval()

# Create opener to access image
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

# Load and preprocess the image
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
url, filename = (sys.argv[1], "images/image.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension

# Get model predictions
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())