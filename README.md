# Van Gogh, but with GAN

ECE C247 Winter 2021 w/ Jonathan Kao

Andy Cai, Stephanie Doan, Michael Inoue, Henry Trinh

## Setup

### Virtual Environment

Install virtualenv
```
pip3 install virtualenv
```

Create virtualenv
```
python3 -m virtualenv env
``` 

Activate
```
source env/bin/activate
```

Install dependencies
```
pip3 install -r requirements.txt
```

Save dependencies  
```
pip3 freeze > requirements.txt
```

Deactivate 
```
deactivate
```

### Dataset
Install [Kaggle dataset](https://www.kaggle.com/ipythonx/van-gogh-paintings) and put into `/data` folder.
