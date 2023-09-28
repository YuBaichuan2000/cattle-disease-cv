# Training Document

The Training is quite simple that by running `train.py` for training and `eval.py` for testing. For further instructions by change the hyperparameters please refer the office document of https://github.com/ashleve/lightning-hydra-template .

## Training
```bash
python src/train.py
```

## Evaluation
```bash
python src/eval.py
```

## MNISTLitModule

The `MNISTLitModule` is a PyTorch Lightning implementation for MNIST classification. PyTorch Lightning offers a structured way of organizing the PyTorch code, which promotes better readability, scalability, and reproducibility.

### Key Components:

#### 1. **Initialization (`__init__` method)**:

The `__init__` method sets up the main components:
- The neural network architecture (`net`).
- Loss function, in this case, `CrossEntropyLoss`.
- Metrics for calculating and averaging accuracy and loss across batches during training, validation, and testing.
- A metric (`val_acc_best`) to track the best validation accuracy across epochs.

The `save_hyperparameters` method ensures that the hyperparameters passed to the initializer are saved, making them accessible through the `self.hparams` attribute.

#### 2. **Forward Pass (`forward` method)**:

This method defines the forward propagation of the input tensor `x` through the neural network.

#### 3. **Train Loop (`training_step` method)**:

Defines the training process for each batch of data:
- Calculates loss and predictions using the shared `step` method.
- Updates and logs the training loss and accuracy metrics.
- Returns a dictionary containing the batch loss, predictions, and ground-truth labels.

#### 4. **Validation Loop (`validation_step` method)**:

Defines the validation process for each batch:
- Calculates loss and predictions similarly to the training loop.
- Applies the `mask_outputs` function on predictions and ground-truth labels.
- Updates and logs the validation loss and accuracy metrics.
- Returns a dictionary with the same structure as in the training loop.

#### 5. **Test Loop (`test_step` method)**:

Similar to the validation loop, but updates and logs the test loss and accuracy metrics.

#### 6. **Optimizer Configuration (`configure_optimizers` method)**:

Defines which optimizer and learning rate scheduler to use:
- Instantiates the optimizer based on hyperparameters.
- If a scheduler is provided, it defines its properties and attaches it to the optimizer.
- The scheduler monitors the validation loss by default.

---

### Additional Utility Methods:

- **`step` method**: A shared method used by the training, validation, and test loops to calculate loss and predictions for a given batch.
- **`mask_outputs` method**: Modifies the predicted outputs, probably to adjust the class predictions based on some criteria. This method's exact purpose should be clarified further based on the broader context.
- **`on_train_start` method**: Ensures the metric `val_acc_best` is reset before the start of training.
- **`training_epoch_end`, `validation_epoch_end`, `test_epoch_end` methods**: These are placeholder methods that can be used to perform some operations at the end of each epoch. Currently, only `validation_epoch_end` logs the best validation accuracy across epochs.

---

### References:

[LightningModule Documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)


## TIMMBackbone

The `TIMMBackbone` class provides a convenient way to instantiate and utilize models from the TIMM (Torch Image Models) library. TIMM offers a collection of pre-trained and SOTA deep learning models for computer vision tasks.

---

### Key Components:

#### 1. **Initialization (`__init__` method)**:

The `__init__` method is responsible for setting up the model:
- **model_name**: Specifies the model's name from the TIMM library, defaulting to 'efficientnet_b0'.
- **pretrained**: Determines if the model should be initialized with pre-trained weights or from scratch. Defaults to `True`, meaning the model will use pre-trained weights when instantiated.
- **num_classes**: Indicates the number of output classes for the model. It defaults to 3, but can be adjusted based on the specific problem at hand.

The `timm.create_model` function is used to instantiate the desired model based on the provided parameters.

#### 2. **Forward Pass (`forward` method)**:

This method defines the forward propagation of the input tensor `x` through the model.

---

### Usage:

To use the `TIMMBackbone`, one can simply instantiate it by providing the desired model name and other optional parameters if necessary:

```python
backbone = TIMMBackbone(model_name='efficientnet_b1', pretrained=True, num_classes=5)
output = backbone(input_tensor)
```

---

### References:

The TIMM library provides a wide range of models and has become popular in the deep learning community for its flexibility and comprehensive collection. More information can be found in the official TIMM repository.
