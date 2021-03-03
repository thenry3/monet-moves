# Van Gogh, but with GAN

ECE C247 Winter 2021 w/ Jonathan Kao

## Setup

### Dataset
Download [Kaggle dataset](https://www.kaggle.com/ipythonx/van-gogh-paintings) and put into `/data` folder.

### Virtual Environment

Install and run virtualenv, install dependencies.
```
pip3 install virtualenv
python3 -m virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

Save dependencies and deactivate.
```
pip3 freeze > requirements.txt
deactivate
```
