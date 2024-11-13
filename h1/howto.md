# Notebook configs

This notebook trains first a classifier based on a base model (freezing its weights) and then fine tuning the model by unfreezing some base models's layers (https://www.tensorflow.org/tutorials/images/transfer_learning).

As the guide states:
```
Note: This should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable. If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will forget what it has learned.
```

## Image size
The default input image size is `96x96x3` but you can rescale the image (e.g. enlarge or center crop) by changing [this](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C7:L2) size.

This size is read from the model input layer so changing this will change the image going through the network (automatically with the augmentation layer where a center crop layer is being used). If left to `96` nothing will change.

## Apply class weights
Control this flag [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C5:L2)

## Dataset splitting
You can control the splitting percentages [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C9:L2)

## Training params
Batch size and epochs are defined [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C16:L2)

## Optimizer configuration
Optimizer can be [configured](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C17:L1) separately for classification training and base model fine tuning. Separate learning rates are applied.
You can define an exponential decay policy by defining a float number [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C17:L12).

Exponential learning rate can be used both on the first training as the second one. By default if [this](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C17:L12) is not `None` the exponential decay is applied only on the first training.
If you want to enable it also when fine tuning the model please call:
```
get_optimizer(is_fine_tuning = True, use_decay_fine_tuning = True)
```

Custom weight_decay values and momentum can be specified as `kwargs`:
```
# Example custom momentum for SGD
optimizer=get_optimizer(momentum=0.89)

# Example custom weight_decay for SGD
optimizer=get_optimizer(weight_decay=1e-4)
```

This API is called [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C30:L10) for classifier training and [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C30:L29) for fine tuning. Please modify the params based on your needs.

## Classification layers
A variable number of dropout layers followed by dense layers cane be defined [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C18:L3).

Make sure the first dense layer size is compatible with the size which comes from the base model's GAP.

## Base model selection
Can be performed [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C20:L10).

## Trained classifier loading
To speed up different testings when you have to make different tests but only the second training changes (fine tuning) you can load a pre trained classification model (with base model's layers frozen, dense layers trained).
Control the flags and model path [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C21:L2).

By default this checkpoint is alway saved so you will always find this intermediate model.

## Select fine tuning layers
Add every base model's layers you want to fine tune [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C22:L2). To find the layers name you can [display](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C29:L1) the model.

We adopt a name based filter as some layers have `0` trainable params so defining the number of layers to unfreeze may not always unfreeze the wanted number of layers.

## Augmentation
You can build the desired augmentation pipeline [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C24:L2).

## Start the training
The training and fine tuning can be started [here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C30:L1). It's ready to use and the only thing you may change are the optimizer configurations (as shown above).

Classifier training and fine tuning are in the same block to allow you to sleep 8h straight.

## Plot training history
[here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C31:L23).

## Evaluate model
[here](https://vscode.dev/github/mattteochen/polimi-anndl/blob/main/h1/main.ipynb#C35:L1).