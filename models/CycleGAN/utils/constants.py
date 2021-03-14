# Root directory for dataset
data_dir = './../../data'

# Root directory for saved models
saved_models_dir = './../saved_models'

# Batch size during training
batch_size = 1

# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 256

# Number of training epochs
num_epochs = 1

# Learning rate for optimizers
lr = {
    'G': 0.0002,
    'D': 0.0002
}

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of workers for dataloader
workers = 4
