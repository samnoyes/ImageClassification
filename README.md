# ImageClassification

## About
This image classification tool uses HuggingFace's [timm library](https://huggingface.co/docs/timm/index) with a [ResNeXt model](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/) to perform image classification.

## Installation

1. Ensure you have Python installed

2. Clone this repository to your local machine using the following command:

`git clone https://github.com/samnoyes/ImageClassification.git`

3. Install the required dependencies. In terminal: navigate to the project directory and run the following command:

`pip install -r requirements.txt` (`pip3` if you are using python3)

This will install the necessary packages.

## Usage

1. Find an image you would like to classify, and edit image_classification.py with the URL of your image (code comments show where to edit)

2. `python3 image_classification.py` (or `python3` depending on your version)

### Sample Output:

~~~
lion 0.9996147155761719
chow 0.000105000362964347
tiger 5.202060492592864e-05
leopard 3.0612063710577786e-05
cheetah 2.1639469196088612e-05
~~~

## Credit

Followed a tutorial from the HuggingFace docs here: https://huggingface.co/docs/timm/models/resnext