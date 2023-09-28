# Data Modules

We following the best practice in PyTorch Lighting that using datamodule to organize the dataset and dataloader you can refer source file [cow_datamodule.py](../code/src/datamodules/cow_datamodule.py) and config file [cowv3.yaml](..%2Fcode%2Fconfigs%2Fdatamodule%2Fcowv3.yaml).

## PyTorch Dataset

```python
class CowCustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.len = len(self.img_labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = cv2.imread(img_path)
        label = int(self.img_labels.iloc[idx, 2])
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

`CowCustomImageDataset` is a custom dataset class for loading images and their associated labels. This class is designed to be compatible with PyTorch's Dataset interface. The primary purpose is to fetch cow images from a specified directory and their corresponding labels from an annotations CSV file.

### Initialization (`__init__`):

### Parameters:

- `annotations_file`: Path to the CSV file containing image annotations. The CSV file is expected to have at least two columns: the first column for image filenames and the second column for image labels.

- `img_dir`: Directory path where the images are stored.

- `transform`: (Optional) Transformations to be applied on the images. It is expected that the transformations are compatible with the `albumentations` library format, where the output should be a dictionary with a key "image" mapping to the transformed image.

- `target_transform`: (Optional) Transformations to be applied on the labels.

### Attributes:

- `img_labels`: DataFrame loaded from the `annotations_file` CSV, containing image filenames and their corresponding labels.

- `img_dir`: Directory path where the images are stored.

- `transform`: Transformations to be applied on the images.

- `target_transform`: Transformations to be applied on the labels.

- `len`: Total number of images/labels in the dataset.

### Length Method (`__len__`):

Returns the total number of images/labels present in the dataset.

### Item Getter Method (`__getitem__`):

### Parameters:

- `idx`: Index of the image/label to be fetched.

### Returns:

- `image`: Image at the given index, read using OpenCV. Any specified transformations will be applied to the image before returning.

- `label`: Corresponding label of the image at the given index. Any specified target transformations will be applied to the label before returning.

## Remarks:

- The code has commented-out sections where the image was originally read using PIL and converted to an array. This approach is replaced by directly reading the image using OpenCV.

- The image is read in BGR format by default when using OpenCV's `imread`. If you wish to work with images in RGB format (especially if you plan to visualize them using tools that expect RGB format), you may want to uncomment the line `image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` to convert the BGR image to RGB.

## Example Usage:

```python
from torch.utils.data import DataLoader

# Initialize dataset
dataset = CowCustomImageDataset(annotations_file='path_to_annotations.csv', img_dir='path_to_images')

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterating through the DataLoader
for images, labels in data_loader:
    # Your training or evaluation code here
    pass
```
## PyTorch LightningDataModule

## Overview

The `CowDataModule` is a PyTorch Lightning `LightningDataModule` implementation specifically designed for managing cow image datasets. By encapsulating data loading, preparation, splitting, and augmentation logic, the module facilitates cleaner and more scalable PyTorch Lightning projects.

## Methods

1. **prepare_data**:
   - Called only once for downloading data and/or performing one-time preprocessing.
   - Don't set state in this method.

2. **setup**:
   - Load the dataset and create splits for training, validation, and testing.
   - Called on each GPU/TPU (in DDP).

3. **train_dataloader**:
   - Returns the training DataLoader.

4. **val_dataloader**:
   - Returns the validation DataLoader.

5. **test_dataloader**:
   - Returns the testing DataLoader.

6. **teardown**:
   - Clean-up method, called after `fit` or `test` ends.

7. **state_dict**:
   - Return additional state to save in a checkpoint.

8. **load_state_dict**:
   - Operations to perform when loading a checkpoint.

## Initialization Parameters

- **data_dir** (str): Directory where the dataset is stored. Default: "data/".
- **train_val_test_split** (Tuple[int, int, int]): Number of samples in the train, validation, and test splits, respectively.
- **batch_size** (int): Number of samples per batch. Default: 64.
- **num_workers** (int): Number of subprocesses to use for data loading. Default: 0.
- **pin_memory** (bool): Whether to copy Tensors into CUDA pinned memory before returning them. Useful when training on GPUs. Default: False.
- **annotations_file** (str): Path to the CSV containing image annotations. Default: "data/annotations.csv".
- **k** (int): The fold number for k-fold cross-validation. `-1` disables k-fold splitting. Default: 1.
- **n_splits** (int): Total number of folds in k-fold cross-validation. Default: 5.

## Data Transformations

The module uses `albumentations` for data augmentation and transformation. The following augmentations are defined:

- Horizontal flip.
- Vertical flip.
- Shift, scale, rotate.
- Random resized crop.
- Motion blur.
- Random brightness and contrast adjustment.
- Gaussian noise.
- Rotation.
- Resize to 224x224.
- Normalize with predefined mean and standard deviation values.
- Convert to PyTorch tensor.

## Remarks

- The `prepare_data` method currently contains commented-out lines meant for the MNIST dataset. It appears this code was adapted from an MNIST example, and these lines should be adjusted or removed depending on the actual data source for cow images.
- In the `setup` method, the data is split either using a simple random split or k-fold cross-validation based on the value of the `k` parameter.
- The `test_dataloader` currently returns the same data as the `val_dataloader`.

## Example Usage

```python
from pytorch_lightning import Trainer

# Initialize DataModule
data_module = CowDataModule(data_dir='path_to_data', annotations_file='path_to_annotations.csv')

# Initialize model (assuming you have a LightningModule called CowModel)
model = CowModel()

# Initialize Lightning Trainer and train model
trainer = Trainer(max_epochs=10)
trainer.fit(model, data_module)
```


## CowDataModule Documentation

The `CowDataModule` is a PyTorch Lightning `LightningDataModule` implementation specifically designed for managing cow image datasets. By encapsulating data loading, preparation, splitting, and augmentation logic, the module facilitates cleaner and more scalable PyTorch Lightning projects.

### Methods

1. **prepare_data**:
   - Called only once for downloading data and/or performing one-time preprocessing.
   - Don't set state in this method.

2. **setup**:
   - Load the dataset and create splits for training, validation, and testing.
   - Called on each GPU/TPU (in DDP).

3. **train_dataloader**:
   - Returns the training DataLoader.

4. **val_dataloader**:
   - Returns the validation DataLoader.

5. **test_dataloader**:
   - Returns the testing DataLoader.

6. **teardown**:
   - Clean-up method, called after `fit` or `test` ends.

7. **state_dict**:
   - Return additional state to save in a checkpoint.

8. **load_state_dict**:
   - Operations to perform when loading a checkpoint.

### Initialization Parameters

- **data_dir** (str): Directory where the dataset is stored. Default: "data/".
- **train_val_test_split** (Tuple[int, int, int]): Number of samples in the train, validation, and test splits, respectively.
- **batch_size** (int): Number of samples per batch. Default: 64.
- **num_workers** (int): Number of subprocesses to use for data loading. Default: 0.
- **pin_memory** (bool): Whether to copy Tensors into CUDA pinned memory before returning them. Useful when training on GPUs. Default: False.
- **annotations_file** (str): Path to the CSV containing image annotations. Default: "data/annotations.csv".
- **k** (int): The fold number for k-fold cross-validation. `-1` disables k-fold splitting. Default: 1.
- **n_splits** (int): Total number of folds in k-fold cross-validation. Default: 5.

### Data Transformations

The module uses `albumentations` for data augmentation and transformation. The following augmentations are defined:

- Horizontal flip.
- Vertical flip.
- Shift, scale, rotate.
- Random resized crop.
- Motion blur.
- Random brightness and contrast adjustment.
- Gaussian noise.
- Rotation.
- Resize to 224x224.
- Normalize with predefined mean and standard deviation values.
- Convert to PyTorch tensor.
Certainly! The `setup` method and the use of `KFold` cross-validation in the `CowDataModule` class warrant a closer examination given their crucial roles in data preparation for machine learning tasks.

### `setup` method

#### Purpose:
The `setup` method is responsible for organizing and splitting the dataset into training, validation, and testing datasets. This method ensures that these datasets are prepared once and are ready for subsequent processes.

#### Key Operations in `setup`:

1. **Dataset Initialization**:
   ```python
   dataset = CowCustomImageDataset(self.hparams.annotations_file, self.hparams.data_dir, transform=self.transforms, target_transform=self.target_transforms)
   ```
   - Here, the cow image dataset is loaded into memory by instantiating the `CowCustomImageDataset` class, which has been defined elsewhere.
   - Data transformations (`self.transforms`) and target transformations (`self.target_transforms`) are applied during this initialization.

2. **Data Splitting Based on K-Fold**:
   - K-Fold cross-validation is a method where the dataset is divided into 'k' different subsets (or folds). One of the folds is used for validation, and the remaining 'k-1' folds are used for training. This process is repeated 'k' times, ensuring each fold serves as the validation set once.
   - In this `CowDataModule`, the choice to use K-Fold or not is determined by the `k` attribute. If `k == -1`, K-Fold is not used.
   
   ```python
   if self.hparams.k == -1:
       ...
   else:
       kf = KFold(n_splits=self.hparams.n_splits, shuffle=True, random_state=42)
       all_splits = [k for k in kf.split(dataset)]
       train_indexes, val_indexes = all_splits[self.hparams.k]
       ...
   ```

   - When K-Fold is active (`self.hparams.k != -1`):
     - A `KFold` object (`kf`) is initialized with a specified number of splits (`self.hparams.n_splits`), shuffling enabled, and a fixed random seed for reproducibility.
     - All potential splits are generated using `kf.split(dataset)`.
     - The relevant train and validation indices are then extracted based on the current fold (`self.hparams.k`).

3. **Creation of Train and Validation Datasets**:
   - Using the indices generated either from the K-Fold split or a simple random split, subsets of the main dataset are created for training and validation.
   ```python
   self.data_train = torch.utils.data.Subset(dataset, train_indexes)
   self.data_val = torch.utils.data.Subset(dataset, val_indexes)
   ```

4. **Setting Test Dataset**:
   - Currently, the validation dataset is also being used as the test dataset. This might be a placeholder, and in real scenarios, you might want to have a separate test set.
   ```python
   self.data_test = self.data_val
   ```

### `KFold` Cross-Validation

#### Purpose:
KFold cross-validation is a technique to evaluate the performance of a machine learning model more robustly. By dividing the dataset into 'k' subsets and rotating the validation subset, it reduces the likelihood of a "lucky" or "unlucky" data split influencing model evaluation metrics.

#### Benefits:
1. **Better Model Evaluation**: By using different validation sets, you get a more comprehensive understanding of your model's performance across different parts of the dataset.
2. **Reduced Variance**: Averaging results over 'k' folds can lead to a more reliable estimate of model performance.

#### Drawbacks:
1. **Increased Computational Cost**: Training is done 'k' times, so expect 'k' times the computational cost of a regular single split.

### Remarks

- The `prepare_data` method currently contains commented-out lines meant for the MNIST dataset. It appears this code was adapted from an MNIST example, and these lines should be adjusted or removed depending on the actual data source for cow images.
- In the `setup` method, the data is split either using a simple random split or k-fold cross-validation based on the value of the `k` parameter.
- The `test_dataloader` currently returns the same data as the `val_dataloader`.

### Example Usage

```python
from pytorch_lightning import Trainer

# Initialize DataModule
data_module = CowDataModule(data_dir='path_to_data', annotations_file='path_to_annotations.csv')

# Initialize model (assuming you have a LightningModule called CowModel)
model = CowModel()

# Initialize Lightning Trainer and train model
trainer = Trainer(max_epochs=10)
trainer.fit(model, data_module)
```
