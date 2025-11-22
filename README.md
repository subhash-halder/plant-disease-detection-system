# plant-disease-detection-system

MCA project with Machine learning

## Dataset used

[Kaggle Plant Disease image data set](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Project

There are certain category of images like plant_leaves and there different diseases,
we will train the deep learning model on those data and will categorize any plant
leaves within these category

## Project setup

### Python env setup for mac

Run the below command to setup python env

```bash
# install miniforge first time to install mamba with this we will be able to use tensorflow gpu
brew install --cask miniforge

# Create the env 
mamba create -n tf-gpu python=3.12
```

Run the Below command to activate the env

```bash
eval "$(mamba shell hook --shell zsh)"
mamba activate tf-gpu
```

Command to verify tensorflow installation with GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Tensorflow

[Tensorflow website](https://www.tensorflow.org/)

#### install tensorflow

```bash
python -m pip install --force-reinstall tensorflow-macos tensorflow-metal
```
