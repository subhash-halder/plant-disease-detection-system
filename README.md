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
# Create the env 
python3 -m venv .venv
```

Run the Below command to activate the env

```bash
source .venv/bin/activate
```

### Tensorflow

[Tensorflow website](https://www.tensorflow.org/)

#### install tensorflow

```bash
pip install -r requirement.txt
```

Command to verify tensorflow installation with GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('CPU'))"
```

Run this command to run the jupyter lab to run the code

```bash
pip install jupyterlab
```

```bash
jupyter lab
```
