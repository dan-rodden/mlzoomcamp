# Deep Learning with PyTorch

# Pre-trained Models
- PyTorch is a low-level library compared to keras which is a high-level library built on-top of Tensorflow
- Go to PyTorch `https://docs.pytorch.org/vision/main/models.html` to find models with pre-trained models that can be used as the convolutional layer for a model.
- Going to use `MobileNet V2` for our pre-trained model. One should check the comparison table to look at accuracy of the different models v. number of parameters and size of model.

```python
import torch
import torchvision.models as models
from torchvision import transforms

model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.eval();
```

- Preprocessing is performed to prepare the image for model evaluation:
    - `Resize`: means we make the img be of dimension (256, 256, 3)
    - `CenterCrop`: means we take the newly resized image and take the middle portion of the image. This removes the periphery of the image. It is of size (224, 224, 3)
    - `ToTensor`: Converts the image from an image to a numpy array (or tensor) so that a CNN can understand and manipulate the image.
    - `Normalize()`: Normalize the data so that we have data that is between 0 and 1 rather then 0 to 255 which is how most pictures are composed. 
```python
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

- Now one can apply the preprocessing steps defined above to an image or batch of images. 
    - `preprocess`: applies the preprocessing pipeline defined above to our image
    - `torch.unsqeeze(x, 0)`: turns our data into a batch. Usually we supply a large amount of images for training. 
    - `with torch.no_grad()`: Makes a prediction with `no_grad()` meaning that we are doing this for evaluation only. We are not training a model. We are using the model.
    - `output.shape`: gives shows that we have 1 batch and 1000 predictions for the thousand classes that are defined. 
```python
x = preprocess(img)
batch_t = torch.unsqueeze(x, 0)
batch_t.shape # torch.Size([1, 3, 224, 224]) - unsqueeze turns the single image into a batch. Normally when sending multiple images we should have a batch
with torch.no_grad():
    output = model(batch_t)

output.shape # torch.Size([1, 1000])
```

- If one wants to see which class has the highest prediction values we use `torch.sort()`.
```python
# get the top predictions based on index
_ , indices = torch.sort(output, descending=True)

# download the classes from imagenet
!curl -o ../datasets/imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# get the top classes from torch
with open('../datasets/imagenet_classes.txt', "r") as f:
    categories = [s.strip() for s in f.readlines()]

#top5 indices
top5_indices = indices[0, :5].tolist()
top5_categories = [categories[i] for i in top5_indices]

print("Top 5 predictions")
for i, class_name in enumerate(top5_categories):
    print(f"{i+1}: {class_name}")
```

# Transfer Learning
- Transfer learning reuses a model trained on one task such as image classification for a different task such as clothing classification.
- Approach
    1. Load pre-trained model (feature extractor)
    2. Remove original classification
    3. Freeze convolutional layers
    4. Add custom dense layers for the task
    5. Train only the new layers

- Convolutional Neural Networks are what allow for transfer learning to take place. Here are the key components and workflow.
    1. Convolutional Layer: Extracts features using filters
        - Applies filters (e.g. 3x3, 5x5) to detect patterns
        - Creates feature maps (one per filter)
        - Detects edges, textures, shapes
    2. ReLU Activation: Introduces non-linearity
        - f(x) = max(0, x)
        - Sets negative features to 0
        - Helps network learn complex patterns
    3. Pooling layer: Down sample the feature maps
        - Reduces spatial dimensions
        - Max pooling: takes maximum value in a region
        - Makes features more robust to small translations
    4. Fully Connected (Dense) Layer: Final classification
        - Flattens 2D feature maps to 1D vector
        - Connects to output classes
- Full CNN workflow:
`Input Image → Conv + ReLU → Pooling → Flatten → Dense → Output`


- Now perform `transfer learning` which takes an already pre-trained model, but we tweak the dense layers while keeping the convolution NN that transforms our image into a vector that can be learned by the new dense layers.
- Code for train/validation data for the model in pytorch. This is much more involved then keras:
- General steps:
    - `def __init__ step`:
        - Inherit from the Dataset Class creating a new ClothingDataset Object `ClothingDataset(Dataset)`
        - Define the variables in the class and fill the self.image_paths variable with all of the paths to the images.
        - Fill self.labels with label names stored in self.classes_to_idx
    - `def __getitem__(self, idx)`
        - Takes the image path and converts it ot RGB values then transforms the image.

- Need to create a custom `DataSet Class` to load and process the images. This works for any folder based image processing.
```python
import os
from torch.utils.data import Dataset
from PIL import Image

class ClothingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

```

__Apply Preprocessing to the train and val datasets__
- Perform simple preprocessing similar to preprocessing of the single image above.
- The transformations below correspond to the `preprocess_function=preprocess_input` argument in keras. 
- Eventually the transformation functions shown below will be combined as they are redundant. The same transformation is happening to both the train and validation datasets. Therefore, there is no need for two functions.
```python
from torchvision import transforms

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Simple transforms - just resize and normalize
train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

```
- Use the `ClothingDataset()` object to load our training and validation datasets.
- The `DataLoader()` object ensures that our transformed images are processed as batches
- One should use `shuffle=True` on the `train_dataset` as randomizing the order is better for training as the model goes through all batches during each training epoch.
- For validation `shuffle=False` is better as it makes comparing validation performance easier.

```python
train_dataset = ClothingDataset(
    data_dir='../datasets/clothing-dataset-small-master/train',
    transform=train_transforms
)

val_dataset = ClothingDataset(
    data_dir="../datasets/clothing-dataset-small-master/validation/",
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

# Build the Model
- we want to create a class for building the model.
- We want to freeze the layers for the convolutional neural network which is done with `for param in self.base_model.parameters(): param.requires_grad = False`. This loops through base_model (the CNN layers) and freezes every layer.
- The line `self.output_layer = nn.Linear(1280, num_classes)` initializes a new layer by default. 
- The `forward` method connects the base CNN layer to the new trainable dense layers.
```python
import torch.nn as nn
import torchvision.models as models

class ClothingClassifierMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ClothingClassifierMobileNet, self).__init__()

        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenetv2(weights='IMAGENET1K_V1')

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Remove original classifier
        self.base_model.classifier = nn.Identity()

        # Add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        return x
```

# Train the Model
- This part is the setup for training.
- The line `optimizer.zero_grad()` is crucial in the training loop. In PyTorch gradients are accumulated by default. This means that if you do not zero out the gradients before calculating another batch, the gradients from the previous batch will be added to the gradients of the current batch. This would incorrectly update the model's parameters.
- By calling `optimizer.zero_grad()` you clear out the old gradients, ensuring that the gradients calculated are correct. 
- `model.train()` sets the model for training mode. In training mode, layers like Dropout and BatchNorm behave differently. Dropout layers are actively dropping neurons. BatchNorm layers update their running statistics (mean and variance) based on the current batch.
- `model.eval()` sets the model to evaluation mode. In evaluation mode, Dropout layers are inactive (they pass through all neurons) and BatchNorm layers user their accumulated running statistics instead of the current batch statistics. This ensures consistent behavior during inference and prevents randomness from affecting the evaluation results.

```python
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClothingClassifierMobileNet(num_classes=10)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
```
- Here is the part with the actual training
```python
# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the training data
    for inputs, labels in train_loader:
        # Move data to the specified device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients to prevent accumulation
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        running_loss += loss.item()
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        # Update total and correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Disable gradient calculation for validation
    with torch.no_grad():
        # Iterate over the validation data
        for inputs, labels in val_loader:
            # Move data to the specified device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)

            # Accumulate validation loss
            val_loss += loss.item()
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            # Update total and correct predictions
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # Print epoch results
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
```

# Tuning the Learning Rate
- Learning rate controls how much to update modal weights during training and is a very important hyperparameter. 
- Experimentation approach is to:
    1. try multiple values: `[0.0001, 0.001, 0.01, 0.1]`
    2. Train for a few epochs each
    3. Compare accuracy
    4. Choose the rate with the best performacne and smallest train/val gap

```python
# make the model
def make_model(learning_rate=0.01):
    model = ClothingClassifierMobileNet(num_classes=10)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

# test multiple different learning rates
learning_rates = [0.0001, 0.001, 0.01, 0.1]

for lr in learning_rate:
    print(f"\n=== Learning Rate: {lr} ===")
    model, optimizer = make_model(learning_rate=lr)
    train_and_evaluate(model, optimizer, train_loader, val_loader, criterion num_epochs, device)
```

# Add an Extra Layer 
- Add intermediate dense layers between feature extraction and output which involves creating a new class for the ClothingClassifier. 
```python
class ClothingClassifierMobileNet(nn.Module):
    def __init__(self, size_inner=100, num_classes=10):
        super(ClothingClassifierMobileNet, self).__init__()
        
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.base_model.classifier = nn.Identity()
        
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)  # New inner layer
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
```

- After this we update the `make_model` function.
- We also want to tune the number of dense layers present in the inner layer. 
- We can try a variety of different sizes of inner layers. The rules of thumb are:
    - Larger layers: more capacity and may overfit
    - Smaller layers: Faster but may underfit
```python
def make_model(learning_rate=0.001, size_inner=100):
    model = ClothingClassifierMobileNet(
        num_classes=10,
        size_inner=size_inner
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer
```



# __Notes on the Code__
- Theses notes are a restatement of the above, but with a focus on how the code works rather then theory as shown above.

Notes created with the help of Google Gemini

# PyTorch Custom Dataset: ClothingDataset (Step 1 in the Pytorch Walkthrough)

## Overview
The `ClothingDataset` class is a custom implementation of the PyTorch `Dataset` base class. It is designed to handle image classification tasks where images are organized into folders named after their respective categories.

## Core Imports
* `os`: Handles directory navigation and file path joining.
* `torch.utils.data.Dataset`: The base class that allows this code to work with PyTorch's `DataLoader`.
* `PIL.Image`: Used for opening image files and ensuring they are in the correct color format (RGB).

---

## Detailed Method Breakdown

### 1. `__init__(self, data_dir, transform=None)`
This method initializes the dataset by crawling the file system.
* **Directory Scanning**: It identifies subdirectories as "classes" (labels).
* **Index Mapping**: It maps class names (strings) to integers (0, 1, 2...) because machine learning models require numeric labels.
* **Path Indexing**: It stores the file path for every image and its corresponding numeric label into two parallel lists: `self.image_paths` and `self.labels`.
* **Lazy Loading**: It does **not** load the image data into RAM here; it only stores the "address" of the files.

### 2. `__len__(self)`
* Returns the total number of items in the dataset (`len(self.image_paths)`).
* This tells the training loop or DataLoader how many iterations are required to complete one epoch.

### 3. `__getitem__(self, idx)`
This is triggered when you request a specific item (e.g., `dataset[0]`).
* **Opening**: It opens the image from disk using the path stored at `idx`.
* **Standardization**: Uses `.convert('RGB')` to ensure consistent input (3 channels), even if some source images are Grayscale or RGBA.
* **Transforms**: If any data augmentations or preprocessing steps (like resizing or normalization) were passed to the class, they are applied here.
* **Return**: Returns a tuple: `(image_tensor, numeric_label)`.

---

### 4. `The code`
```python
import os
from torch.utils.data import Dataset
from PIL import Image

class ClothingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 1. Basic Setup
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 2. Class Discovery
        # Finds all folder names in data_dir and sorts them alphabetically.
        self.classes = sorted(os.listdir(data_dir))
        
        # 3. Label Mapping
        # Creates a dictionary like {'dress': 0, 'shirt': 1, 't-shirt': 2}
        # enumerate() gives us the index 'i' for each class 'cls'.
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 4. Path Crawling (The Nested Loop)
        for label_name in self.classes:
            # Create the full path to the subfolder: e.g., 'data/train/dress'
            label_dir = os.path.join(data_dir, label_name)
            
            # Look at every file inside that specific subfolder
            for img_name in os.listdir(label_dir):
                # Store the exact path to the image: e.g., 'data/train/dress/img_1.jpg'
                self.image_paths.append(os.path.join(label_dir, img_name))
                
                # Store the numeric index for that class (e.g., 0 for 'dress')
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label   
```

---

## Why use this structure?
1.  **Memory Efficiency**: By loading images one-by-one inside `__getitem__` rather than all at once in `__init__`, you can train on datasets much larger than your available memory.
2.  **Compatibility**: This structure allows the dataset to be plugged directly into a PyTorch `DataLoader` for automatic batching, shuffling, and multi-process data loading.

---

# Sample Preparation (Step 2 in Pytorch Code Walkthrough)

## PyTorch Image Transformation & Normalization Notes

## 1. Parameters: The "Standard" Settings

### Why is `input_size = 224`?
The value **224x224** is the standard input resolution for many famous Convolutional Neural Networks (CNNs) like ResNet, VGG, and MobileNet.

* **Historical Precedent:** These models were originally trained on the **ImageNet** dataset at this resolution.
* **Efficiency:** 224 is a multiple of 7 and 32, which works well with the "pooling" layers in neural networks that repeatedly halve the image dimensions ($224 \rightarrow 112 \rightarrow 56 \rightarrow 28 \rightarrow 14 \rightarrow 7$).
* **Consistency:** All images in a "batch" must be the exact same dimensions for the math (matrix multiplication) to work.



### ImageNet Normalization: Why and How?
The values `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]` are specific to the millions of images in the ImageNet dataset.

* **How they are calculated:** Researchers calculated the average (mean) and the spread (standard deviation) of every pixel in the entire ImageNet dataset for each color channel (Red, Green, Blue).
* **Why we use them:** If you are using a **pre-trained model** (a model already "taught" on ImageNet), you must normalize your new images using the exact same constants the model saw during its original training. This ensures the "color distribution" of your data matches what the model expects.
* **The Goal:** It centers the data around zero. This prevents "exploding gradients" and helps the model converge (learn) much faster.

---

## 2. Transformation Logic

### What is the point of `train_transforms` and `val_transforms`?
These variables store **objects** that act like a "recipe" or a "pipeline."

* **Storage:** They don't store the images themselves; they store the **instructions** on how to change an image.
* **Separation of Concerns:**
    * `train_transforms`: Often includes "Data Augmentation" (like random flipping) to help the model learn better.
    * `val_transforms`: Usually only includes the bare essentials (Resize and Normalize) to ensure we are testing the model on "clean" data.

### `transforms.Compose()`
Think of this as a **container**. It chains multiple operations together into a single object. When you pass an image to a "Composed" transform, the image goes through the list of functions in order, like an assembly line.


---

## 3. The Individual Operations

| Function | What it does | Why it is necessary |
| :--- | :--- | :--- |
| **`transforms.Resize((224, 224))`** | Changes the height and width of the image to the specified dimensions. | Ensures every image in your dataset has the exact same number of pixels so they can be stacked into batches. |
| **`transforms.ToTensor()`** | 1. Converts a PIL Image or NumPy array to a **PyTorch Tensor**. <br><br> 2. Scales pixel values from **0–255** down to **0.0–1.0**. | Neural networks perform math on Tensors (vectors/matrices). The 0-1 scaling is the first step in numerical stabilization. |
| **`transforms.Normalize()`** | Takes the 0–1 tensor and performs: $$output = \frac{input - mean}{std}$$ | This "standardizes" the data. It ensures that the input features have a mean of 0 and a variance of 1, making the "loss landscape" easier for the optimizer to navigate. |

---

## 4. The code
```python
from torchvision import transforms

input_size = 224

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Simple transforms - just resize and normalize
train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

# PyTorch DataLoaders: (Step 3)

This guide breaks down a common PyTorch pattern for loading data, explaining each line of code in detail.

---

## The Full Code

```python
from torch.utils.data import DataLoader

train_dataset = ClothingDataset(
    data_dir='./clothing-dataset-small/train',
    transform=train_transforms
)

val_dataset = ClothingDataset(
    data_dir='./clothing-dataset-small/validation',
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

## Line-by-Line Explanation

### Line 1: The Import Statement

```python
from torch.utils.data import DataLoader
```

**What this does:** This line imports the `DataLoader` class from PyTorch's utilities.

**Breaking it down:**
- `torch` is the main PyTorch library
- `utils` is a sub-module containing utility tools
- `data` is a sub-module specifically for data handling
- `DataLoader` is the class we're importing

**Why we need it:** The `DataLoader` is a helper that takes your data and serves it up to your model in organized chunks called "batches." Think of it like a waiter who brings your food in courses rather than dumping everything on the table at once.

---

### Lines 3-6: Creating the Training Dataset

```python
train_dataset = ClothingDataset(
    data_dir='./clothing-dataset-small/train',
    transform=train_transforms
)
```

**What this does:** Creates a dataset object that knows where your training images are and how to process them.

**Breaking it down:**

| Parameter | What it means |
|-----------|---------------|
| `ClothingDataset` | A custom class (defined elsewhere) that tells PyTorch how to load your specific data |
| `data_dir='./clothing-dataset-small/train'` | The folder path where your training images live. The `./` means "starting from the current directory" |
| `transform=train_transforms` | A set of image processing steps (like resizing, rotating, etc.) to apply to each image |

**Key Concept - What is a Dataset?**

A dataset in PyTorch is like an organized filing cabinet. It needs to know:
1. How many items it contains
2. How to retrieve any single item by its index (like asking for "item #42")

The `ClothingDataset` class (which would be defined separately) handles these details for your specific clothing images.

**Key Concept - What are Transforms?**

Transforms are processing steps applied to your data. Common examples include:
- Resizing images to a consistent size
- Converting images to PyTorch tensors (the data format PyTorch uses)
- Normalizing pixel values
- Data augmentation (random flips, rotations, etc.) to help the model generalize better

---

### Lines 8-11: Creating the Validation Dataset

```python
val_dataset = ClothingDataset(
    data_dir='./clothing-dataset-small/validation',
    transform=val_transforms
)
```

**What this does:** Creates a separate dataset for validation data.

**Why separate datasets?**

| Dataset | Purpose |
|---------|---------|
| Training | Used to teach the model (the model learns from these examples) |
| Validation | Used to test how well the model is learning (the model never learns from these) |

This separation prevents "cheating" - you want to test your model on data it hasn't seen during training to get an honest assessment of its performance.

**Note:** The validation transforms (`val_transforms`) are typically simpler than training transforms. Training might include random rotations and flips (data augmentation), while validation usually just resizes and normalizes - you want consistent, unmodified images for testing.

---

### Line 13: Creating the Training DataLoader

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**What this does:** Wraps your training dataset in a DataLoader that will feed data to your model.

**Breaking down the parameters:**

#### `train_dataset`
The dataset object we created earlier. The DataLoader will pull data from this.

#### `batch_size=32`
**What it means:** Process 32 images at a time.

**Why use batches?**
1. **Memory:** Loading all 10,000 images at once would crash most computers
2. **Speed:** GPUs are optimized for parallel processing - 32 images at once is faster than 32 separate single images
3. **Learning stability:** Averaging the learning signal across 32 examples produces smoother, more stable training

**Visual example:**
```
If you have 320 images and batch_size=32:
- Batch 1: images 1-32
- Batch 2: images 33-64
- Batch 3: images 65-96
- ... and so on ...
- Batch 10: images 289-320

One complete pass through all 10 batches = 1 "epoch"
```

#### `shuffle=True`
**What it means:** Randomize the order of images before creating batches.

**Why shuffle training data?**
- Prevents the model from learning the order of your data instead of the actual patterns
- If all "shirt" images came first, then all "pants," the model might learn misleading patterns
- Each epoch, the batches will contain different combinations of images

---

### Line 14: Creating the Validation DataLoader

```python
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

**What this does:** Creates a DataLoader for validation data.

**Key difference: `shuffle=False`**

We do NOT shuffle validation data because:
1. We want consistent, reproducible results each time we validate
2. The order doesn't affect learning (we're not training on this data)
3. Makes it easier to track which specific images the model gets wrong

---

## How It All Fits Together

```
┌─────────────────────────────────────────────────────────────┐
│                        YOUR FILES                           │
│  ./clothing-dataset-small/train/       (training images)    │
│  ./clothing-dataset-small/validation/  (validation images)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         DATASET                             │
│  - Knows where files are located                            │
│  - Knows how to load a single image                         │
│  - Applies transforms (resize, normalize, etc.)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       DATALOADER                            │
│  - Groups images into batches                               │
│  - Shuffles order (if requested)                            │
│  - Handles the iteration during training                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       YOUR MODEL                            │
│  - Receives batches of 32 images at a time                  │
│  - Processes them and learns patterns                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Using the DataLoader in Practice

Once created, you typically use DataLoaders in a training loop like this:

```python
for epoch in range(num_epochs):
    # Training phase
    for images, labels in train_loader:
        # 'images' is a batch of 32 images
        # 'labels' is a batch of 32 corresponding labels
        # ... training code here ...
    
    # Validation phase
    for images, labels in val_loader:
        # ... validation code here ...
```

The DataLoader handles all the complexity of:
- Loading images from disk
- Applying transforms
- Grouping into batches
- Shuffling (when enabled)
- Converting to PyTorch tensors

---

## Common Beginner Questions

### Q: Why is batch_size=32 common?
Powers of 2 (16, 32, 64, 128) are popular because they align well with GPU memory architecture. 32 is a reasonable starting point for many tasks.

### Q: What if my dataset isn't evenly divisible by batch_size?
The last batch will simply be smaller. If you have 100 images and batch_size=32, you'll get batches of 32, 32, 32, and 4.

### Q: Do I always need separate train and validation sets?
Yes! Without validation data, you have no way to know if your model is actually learning useful patterns or just memorizing the training data (overfitting).

### Q: What's the difference between validation and test sets?
- **Validation:** Used during training to tune your model and make decisions
- **Test:** Used only once at the very end to get a final, unbiased performance measure

---

## Summary

| Component | Purpose |
|-----------|---------|
| `DataLoader` | Utility class that batches and shuffles your data |
| `Dataset` | Tells PyTorch where your data is and how to load it |
| `batch_size` | How many samples to process at once |
| `shuffle` | Whether to randomize order (True for training, False for validation) |
| `transform` | Image processing steps to apply to each sample |



# Building the PyTorch Model (Step 4)

This guide explains how to build a neural network in PyTorch using **transfer learning** - a powerful technique where we reuse a model that was already trained on millions of images.

---

## The Full Code

```python
import torch.nn as nn
import torchvision.models as models

class ClothingClassifierMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ClothingClassifierMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Remove original classifier
        self.base_model.classifier = nn.Identity()
        
        # Add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        return x
```

---

## Line-by-Line Explanation

### Lines 1-2: The Import Statements

```python
import torch.nn as nn
import torchvision.models as models
```

#### `import torch.nn as nn`

**What this does:** Imports PyTorch's neural network module and gives it the nickname `nn`.

**What's inside `torch.nn`?**
- Building blocks for neural networks (layers, activation functions, loss functions)
- The base class `nn.Module` that all PyTorch models inherit from
- Common layers like `Linear`, `Conv2d`, `BatchNorm`, etc.

**Why `as nn`?** It's a convention that makes code shorter. Instead of writing `torch.nn.Linear`, we can write `nn.Linear`.

#### `import torchvision.models as models`

**What this does:** Imports a collection of pre-trained computer vision models.

**What's available in `torchvision.models`?**

| Model | Description |
|-------|-------------|
| `resnet18`, `resnet50`, etc. | Deep residual networks |
| `vgg16`, `vgg19` | Classic deep networks |
| `mobilenet_v2`, `mobilenet_v3` | Lightweight, mobile-friendly models |
| `efficientnet_b0` - `b7` | Efficient scaling networks |
| And many more... | |

These models come pre-trained on ImageNet (1.2 million images, 1000 categories), so they already "know" how to recognize visual patterns.

---

### Line 4: Defining the Class

```python
class ClothingClassifierMobileNet(nn.Module):
```

**What this does:** Creates a new class (blueprint) for our custom model.

**Breaking it down:**

| Part | Meaning |
|------|---------|
| `class` | Python keyword to define a new class |
| `ClothingClassifierMobileNet` | The name we're giving our model |
| `(nn.Module)` | Our class inherits from PyTorch's base `Module` class |

**Key Concept - What is `nn.Module`?**

`nn.Module` is the parent class for ALL neural networks in PyTorch. By inheriting from it, our class automatically gets:

- The ability to track all learnable parameters
- Methods like `.to(device)` to move the model to GPU
- Methods like `.train()` and `.eval()` to switch modes
- The ability to save and load model weights
- Automatic gradient tracking for backpropagation

**Analogy:** Think of `nn.Module` as a template for building models. It's like inheriting from a "Vehicle" class when building a "Car" - you get wheels, an engine, and brakes for free.

---

### Lines 5-6: The Constructor

```python
def __init__(self, num_classes=10):
    super(ClothingClassifierMobileNet, self).__init__()
```

#### `def __init__(self, num_classes=10):`

**What this does:** Defines the initialization method (constructor) that runs when you create a new instance of the model.

**Parameters:**
- `self` - Reference to the instance being created (required in all Python class methods)
- `num_classes=10` - How many categories we want to classify (default is 10 clothing types)

#### `super(ClothingClassifierMobileNet, self).__init__()`

**What this does:** Calls the constructor of the parent class (`nn.Module`).

**Why is this necessary?** The parent class `nn.Module` has its own initialization code that sets up important internal tracking systems. If we skip this line, our model won't work properly.

**Analogy:** If you're building a custom car (child class) based on a standard vehicle design (parent class), you still need to install the basic vehicle components (engine, wheels) before adding your custom features.

**Modern Python note:** In Python 3, you can also write this as simply `super().__init__()` - both forms work identically.

---

### Lines 8-9: Loading the Pre-trained Model

```python
# Load pre-trained MobileNetV2
self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
```

**What this does:** Downloads and loads MobileNetV2 with pre-trained weights.

**Breaking it down:**

#### `self.base_model`
By assigning to `self.something`, we're storing this as an attribute of our model. This is important because PyTorch will automatically track any `nn.Module` objects stored this way.

#### `models.mobilenet_v2()`
Creates an instance of the MobileNetV2 architecture.

#### `weights='IMAGENET1K_V1'`
Tells PyTorch to load weights that were pre-trained on ImageNet.

| Weights Option | Meaning |
|----------------|---------|
| `weights='IMAGENET1K_V1'` | Load pre-trained weights (V1 version) |
| `weights='IMAGENET1K_V2'` | Load improved pre-trained weights (if available) |
| `weights=None` | Random initialization (no pre-training) |

**Key Concept - What is Transfer Learning?**

```
┌─────────────────────────────────────────────────────────────────┐
│                    WITHOUT TRANSFER LEARNING                    │
│                                                                 │
│  Your small dataset ──► Train from scratch ──► Mediocre results │
│  (1,000 images)         (random start)         (not enough data)│
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     WITH TRANSFER LEARNING                         │
│                                                                    │
│  ImageNet ──► Pre-trained ──► Your data ──► Fine-tune ──► Great    │
│  (1.2M imgs)   MobileNet      (1,000 imgs)   (small tweaks) results│
└────────────────────────────────────────────────────────────────────┘
```

The pre-trained model has already learned to recognize:
- Edges and textures (early layers)
- Shapes and patterns (middle layers)
- Complex features like eyes, wheels, fabric patterns (later layers)

We're borrowing all that knowledge instead of learning it from scratch!

**Key Concept - Why MobileNetV2?**

| Property | Benefit |
|----------|---------|
| Lightweight | Only ~3.5 million parameters (vs 25M for ResNet50) |
| Fast | Designed for mobile devices, runs quickly |
| Efficient | Good accuracy despite small size |
| Well-tested | Widely used and reliable |

---

### Lines 11-13: Freezing the Base Model

```python
# Freeze base model parameters
for param in self.base_model.parameters():
    param.requires_grad = False
```

**What this does:** Prevents the pre-trained weights from being updated during training.

**Breaking it down:**

#### `self.base_model.parameters()`
Returns an iterator over all the learnable parameters (weights and biases) in MobileNetV2. There are millions of these numbers!

#### `param.requires_grad = False`
Tells PyTorch: "Don't calculate gradients for this parameter, and don't update it during training."

**Key Concept - What is `requires_grad`?**

Every parameter in PyTorch has a `requires_grad` flag:

| Value | Meaning |
|-------|---------|
| `True` | PyTorch will track operations, calculate gradients, and update this parameter during training |
| `False` | Parameter is "frozen" - it stays exactly as it is |

**Key Concept - Why Freeze?**

```
┌────────────────────────────────────────────────────────────────┐
│                     FROZEN (requires_grad=False)               │
│                                                                │
│  Pros:                          Cons:                          │
│  • Much faster training         • Can't adapt pre-trained      │
│  • Less memory usage              features to your specific    │
│  • Prevents overfitting           domain                       │
│  • Works well with small data                                  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                   UNFROZEN (requires_grad=True)                │
│                                                                │
│  Pros:                          Cons:                          │
│  • Model can fully adapt        • Slower training              │
│  • Potentially higher accuracy  • Risk of overfitting          │
│  • Better for large datasets    • Needs more data              │
└────────────────────────────────────────────────────────────────┘
```

For small datasets (like most clothing datasets), freezing is usually the right choice.

---

### Lines 15-16: Removing the Original Classifier

```python
# Remove original classifier
self.base_model.classifier = nn.Identity()
```

**What this does:** Replaces MobileNetV2's original classification layer with a "do nothing" layer.

**Why do we need to do this?**

MobileNetV2 was trained to classify 1000 ImageNet categories (like "golden retriever", "sports car", "pizza"). Its final layer outputs 1000 values.

We want to classify clothing into 10 categories, so we need to replace that final layer.

**Key Concept - What is `nn.Identity()`?**

`nn.Identity()` is a layer that does absolutely nothing - it just passes its input straight through unchanged.

```
Input: [1, 2, 3, 4, 5]
           │
           ▼
    ┌─────────────┐
    │ nn.Identity │
    └─────────────┘
           │
           ▼
Output: [1, 2, 3, 4, 5]  (exactly the same!)
```

**Why use `nn.Identity()` instead of just deleting the layer?**
- Keeps the model structure intact
- Avoids potential errors from missing expected layers
- Clean and explicit way to "disable" a layer

---

### Lines 18-20: Adding Custom Layers

```python
# Add custom layers
self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
self.output_layer = nn.Linear(1280, num_classes)
```

#### `nn.AdaptiveAvgPool2d((1, 1))`

**What this does:** Creates a pooling layer that reduces any spatial dimensions down to 1×1.

**Key Concept - What is Pooling?**

Pooling reduces the size of feature maps by summarizing regions:

```
Before pooling (4×4):          After AdaptiveAvgPool2d((1,1)):
┌────┬────┬────┬────┐         
│ 1  │ 2  │ 3  │ 4  │         ┌──────┐
├────┼────┼────┼────┤         │ 8.5  │  (average of all 16 values)
│ 5  │ 6  │ 7  │ 8  │   ───►  └──────┘
├────┼────┼────┼────┤         
│ 9  │ 10 │ 11 │ 12 │         Output size: 1×1
├────┼────┼────┼────┤         
│ 13 │ 14 │ 15 │ 16 │         
└────┴────┴────┴────┘         
```

**Why "Adaptive"?**

| Type | Behavior |
|------|----------|
| Regular pooling | You specify kernel size (e.g., 2×2), output size depends on input |
| Adaptive pooling | You specify desired output size, kernel adjusts automatically |

`AdaptiveAvgPool2d((1, 1))` means "no matter the input size, give me a 1×1 output."

#### `nn.Linear(1280, num_classes)`

**What this does:** Creates a fully connected (dense) layer that maps 1280 features to our class predictions.

```
Input: 1280 features (from MobileNetV2)
         │
         ▼
  ┌─────────────────┐
  │   nn.Linear     │
  │  1280 → 10      │
  │                 │
  │ (1280 × 10 =    │
  │  12,800 weights │
  │  + 10 biases)   │
  └─────────────────┘
         │
         ▼
Output: 10 values (one score per clothing class)
```

**Why 1280?**

MobileNetV2's feature extractor outputs 1280 channels. This is a fixed architectural choice made by the MobileNetV2 designers - we need to match it.

**Why `num_classes`?**

We need one output value per category we're classifying:
- Output 0: Score for "t-shirt"
- Output 1: Score for "pants"
- Output 2: Score for "dress"
- ... and so on

---

### Lines 22-27: The Forward Method

```python
def forward(self, x):
    x = self.base_model.features(x)
    x = self.global_avg_pooling(x)
    x = torch.flatten(x, 1)
    x = self.output_layer(x)
    return x
```

**What this does:** Defines how data flows through the model.

**Key Concept - What is the Forward Method?**

The `forward` method is called automatically when you pass data through your model:

```python
# When you write:
output = model(input_image)

# PyTorch actually calls:
output = model.forward(input_image)
```

Every `nn.Module` subclass MUST define a `forward` method - it's the heart of your model.

#### Line-by-line data flow:

**Step 1:** `x = self.base_model.features(x)`

```
Input x: Image tensor
Shape: [batch_size, 3, 224, 224]
       (batch, RGB channels, height, width)
                    │
                    ▼
         ┌─────────────────────┐
         │  MobileNetV2        │
         │  Feature Extractor  │
         │                     │
         │  (Convolutions,     │
         │   BatchNorm,        │
         │   ReLU activations) │
         └─────────────────────┘
                    │
                    ▼
Output x: Feature maps
Shape: [batch_size, 1280, 7, 7]
       (batch, channels, height, width)
```

The `.features` attribute of MobileNetV2 contains all the convolutional layers that extract visual features.

**Step 2:** `x = self.global_avg_pooling(x)`

```
Input x: [batch_size, 1280, 7, 7]
                    │
                    ▼
         ┌─────────────────────┐
         │ AdaptiveAvgPool2d   │
         │      (1, 1)         │
         └─────────────────────┘
                    │
                    ▼
Output x: [batch_size, 1280, 1, 1]
```

Each of the 1280 feature maps (7×7) is reduced to a single average value (1×1).

**Step 3:** `x = torch.flatten(x, 1)`

```
Input x: [batch_size, 1280, 1, 1]
                    │
                    ▼
         ┌─────────────────────┐
         │   torch.flatten     │
         │   (starting dim 1)  │
         └─────────────────────┘
                    │
                    ▼
Output x: [batch_size, 1280]
```

**What does `flatten` do?**

Converts multi-dimensional data into a 1D vector (per sample in the batch).

| Parameter | Meaning |
|-----------|---------|
| `x` | The tensor to flatten |
| `1` | Start flattening from dimension 1 (keep batch dimension intact) |

Example:
```
Before: shape [32, 1280, 1, 1]  (32 samples, each is 1280×1×1)
After:  shape [32, 1280]        (32 samples, each is a 1280-length vector)
```

**Step 4:** `x = self.output_layer(x)`

```
Input x: [batch_size, 1280]
                    │
                    ▼
         ┌─────────────────────┐
         │     nn.Linear       │
         │    1280 → 10        │
         └─────────────────────┘
                    │
                    ▼
Output x: [batch_size, 10]
         (10 scores per image)
```

**Step 5:** `return x`

Returns the final predictions. Each image now has 10 scores (one per clothing class).

---

## Complete Data Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                                 │
│                    Shape: [32, 3, 224, 224]                         │
│              (32 images, 3 color channels, 224×224 pixels)          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MOBILENETV2 FEATURES (FROZEN)                     │
│                                                                     │
│   • Multiple convolutional layers                                   │
│   • Extracts visual patterns (edges, textures, shapes)              │
│   • Pre-trained on 1.2 million ImageNet images                      │
│   • Parameters are frozen (not updated during training)             │
│                                                                     │
│                    Output: [32, 1280, 7, 7]                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GLOBAL AVERAGE POOLING                           │
│                                                                     │
│   • Reduces each 7×7 feature map to a single value                  │
│   • Makes model robust to input position variations                 │
│                                                                     │
│                    Output: [32, 1280, 1, 1]                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FLATTEN                                    │
│                                                                     │
│   • Reshapes 3D tensor to 1D vector (per sample)                    │
│                                                                     │
│                      Output: [32, 1280]                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER (TRAINABLE)                         │
│                                                                     │
│   • Fully connected layer: 1280 inputs → 10 outputs                 │
│   • These parameters ARE updated during training                    │
│   • Learns to map features to clothing categories                   │
│                                                                     │
│                       Output: [32, 10]                              │
│              (32 images × 10 class scores each)                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PREDICTIONS                                  │
│                                                                     │
│   Image 1: [0.1, 0.05, 0.7, 0.02, ...]  → Class 2 (highest score)  │
│   Image 2: [0.8, 0.1, 0.02, 0.01, ...]  → Class 0 (highest score)  │
│   ...                                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How to Use This Model

```python
# 1. Create an instance of the model
model = ClothingClassifierMobileNet(num_classes=10)

# 2. Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 3. Set to training or evaluation mode
model.train()  # For training (enables dropout, etc.)
model.eval()   # For inference (disables dropout, etc.)

# 4. Pass images through the model
images = torch.randn(32, 3, 224, 224)  # Fake batch of 32 images
images = images.to(device)
predictions = model(images)  # Shape: [32, 10]

# 5. Get the predicted class
predicted_classes = predictions.argmax(dim=1)  # Shape: [32]
```

---

## Trainable vs Frozen Parameters

Here's a summary of what gets trained:

| Component | Trainable? | Parameters |
|-----------|------------|------------|
| MobileNetV2 features | ❌ Frozen | ~2.2 million |
| Global average pooling | N/A | 0 (no parameters) |
| Output layer | ✅ Trainable | 1280 × 10 + 10 = 12,810 |

**Total trainable parameters:** ~12,810 (out of ~2.2 million total)

This is the power of transfer learning - we only need to train a tiny fraction of the model!

---

## Common Beginner Questions

### Q: Why use `self.` before each layer?
Storing layers as `self.something` registers them with PyTorch's parameter tracking system. Without `self.`, PyTorch won't know about the layer and won't save/load its weights properly.

### Q: Can I unfreeze the base model later?
Yes! A common technique is to:
1. First train with frozen base (fast, learns custom classifier)
2. Then unfreeze and train everything with a very small learning rate (fine-tuning)

```python
# To unfreeze:
for param in model.base_model.parameters():
    param.requires_grad = True
```

### Q: Why don't we apply softmax in the forward method?
PyTorch's loss function `CrossEntropyLoss` expects raw scores (logits) and applies softmax internally. Applying softmax twice would break training.

### Q: What if I want to classify more or fewer classes?
Just change `num_classes` when creating the model:
```python
model = ClothingClassifierMobileNet(num_classes=5)   # 5 categories
model = ClothingClassifierMobileNet(num_classes=100) # 100 categories
```

### Q: Why 224×224 input size?
MobileNetV2 was designed and trained on 224×224 images. While it can handle other sizes, 224×224 typically gives the best results.

---

## Summary

| Concept | Purpose |
|---------|---------|
| `nn.Module` | Base class that provides neural network functionality |
| `__init__` | Defines the model's layers and structure |
| `forward` | Defines how data flows through the layers |
| Transfer learning | Reuse knowledge from a model trained on millions of images |
| Freezing | Prevent pre-trained weights from changing |
| `nn.Identity()` | "No-op" layer to disable original classifier |
| `nn.AdaptiveAvgPool2d` | Reduce spatial dimensions to fixed size |
| `nn.Linear` | Fully connected layer for final classification |



# Training the PyTorch Model (Step 5)


This guide explains how to set up PyTorch for training a neural network. We'll go into deep detail on the **optimizer** and **loss function (criterion)** - two of the most important concepts in deep learning.

*Note: This code is from the MLZoomcamp course.*

---

## The Full Code

```python
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClothingClassifierMobileNet(num_classes=10)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
```

---

## Line-by-Line Explanation

### Lines 1-2: The Import Statements

```python
import torch
import torch.optim as optim
```

#### `import torch`

**What this does:** Imports the main PyTorch library.

**What we use it for here:**
- `torch.device()` - to specify CPU or GPU
- `torch.cuda.is_available()` - to check if a GPU is available

#### `import torch.optim as optim`

**What this does:** Imports PyTorch's optimization module.

**What's inside `torch.optim`?**

| Optimizer | Description |
|-----------|-------------|
| `optim.SGD` | Stochastic Gradient Descent (the classic) |
| `optim.Adam` | Adaptive Moment Estimation (very popular) |
| `optim.AdamW` | Adam with weight decay fix |
| `optim.RMSprop` | Root Mean Square Propagation |
| `optim.Adagrad` | Adaptive Gradient |
| And more... | |

Optimizers are algorithms that update model weights to minimize the loss function. We'll explore this in detail later.

---

### Lines 4: Device Selection

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**What this does:** Creates a device object that specifies where computations will run.

**Breaking it down:**

#### `torch.cuda.is_available()`
Returns `True` if a CUDA-capable GPU is available, `False` otherwise.

#### The conditional expression
```python
"cuda" if torch.cuda.is_available() else "cpu"
```
This is Python's ternary operator:
- If GPU available → use `"cuda"`
- If no GPU → use `"cpu"`

#### `torch.device(...)`
Creates a device object that PyTorch uses to know where to put tensors and models.

**Key Concept - Why GPUs?**

| Hardware | Cores | Best For |
|----------|-------|----------|
| CPU | 4-16 powerful cores | Sequential tasks, general computing |
| GPU | 1000s of simple cores | Parallel math (perfect for neural networks!) |

```
Training on CPU:                    Training on GPU:
┌─────────────────────┐            ┌─────────────────────┐
│  Process images     │            │  Process images     │
│  one calculation    │            │  THOUSANDS of       │
│  at a time          │            │  calculations       │
│                     │            │  SIMULTANEOUSLY     │
│  Time: 10 hours     │            │  Time: 30 minutes   │
└─────────────────────┘            └─────────────────────┘
```

**Common device values:**

| Device | Meaning |
|--------|---------|
| `"cpu"` | Use the CPU |
| `"cuda"` | Use the default GPU |
| `"cuda:0"` | Use the first GPU |
| `"cuda:1"` | Use the second GPU (if you have multiple) |
| `"mps"` | Use Apple Silicon GPU (M1/M2 Macs) |

---

### Lines 6-7: Creating and Moving the Model

```python
model = ClothingClassifierMobileNet(num_classes=10)
model.to(device)
```

#### Line 6: `model = ClothingClassifierMobileNet(num_classes=10)`

**What this does:** Creates an instance of our custom model class.

**What happens when this line runs:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL INSTANTIATION                          │
│                                                                 │
│  1. Python calls __init__(self, num_classes=10)                 │
│                                                                 │
│  2. super().__init__() sets up nn.Module internals              │
│                                                                 │
│  3. MobileNetV2 is downloaded (first time) or loaded from cache │
│                                                                 │
│  4. Pre-trained weights are loaded into the base model          │
│                                                                 │
│  5. Base model parameters are frozen (requires_grad = False)    │
│                                                                 │
│  6. Custom layers are created:                                  │
│     • global_avg_pooling = AdaptiveAvgPool2d((1, 1))            │
│     • output_layer = Linear(1280, 10)                           │
│                                                                 │
│  7. Model is ready (currently on CPU by default)                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Concept - What IS a Model Variable?**

The `model` variable holds a Python object that contains:

```
model (ClothingClassifierMobileNet object)
│
├── .base_model (MobileNetV2)
│   ├── .features (Sequential of conv layers)
│   │   ├── Layer 0: ConvBNActivation
│   │   ├── Layer 1: InvertedResidual
│   │   ├── ... (many more layers)
│   │   └── Layer 18: ConvBNActivation
│   │
│   └── .classifier (now nn.Identity - does nothing)
│
├── .global_avg_pooling (AdaptiveAvgPool2d)
│
├── .output_layer (Linear)
│   ├── .weight (tensor of shape [10, 1280]) ─── TRAINABLE
│   └── .bias (tensor of shape [10]) ─────────── TRAINABLE
│
└── Methods:
    ├── .forward(x) - defines data flow
    ├── .parameters() - returns all parameters
    ├── .to(device) - moves model to CPU/GPU
    ├── .train() - sets training mode
    ├── .eval() - sets evaluation mode
    └── ... many more inherited from nn.Module
```

#### Line 7: `model.to(device)`

**What this does:** Moves all model parameters (weights and biases) to the specified device.

**Why is this necessary?**

In PyTorch, tensors and models must be on the SAME device to interact:

```
❌ WRONG - Will crash:
┌─────────┐     ┌─────────┐
│  Model  │     │  Data   │
│  (GPU)  │  +  │  (CPU)  │  = RuntimeError!
└─────────┘     └─────────┘

✅ CORRECT - Works:
┌─────────┐     ┌─────────┐
│  Model  │     │  Data   │
│  (GPU)  │  +  │  (GPU)  │  = Success!
└─────────┘     └─────────┘
```

**What `.to(device)` moves:**

| What Gets Moved | Example |
|-----------------|---------|
| All weight tensors | `output_layer.weight` |
| All bias tensors | `output_layer.bias` |
| Buffers (like BatchNorm running stats) | `running_mean`, `running_var` |

**Note:** `.to(device)` modifies the model in-place AND returns the model, so you'll often see:
```python
model = model.to(device)  # Also valid, same result
```

---

## Deep Dive: The Optimizer

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### What is an Optimizer?

An optimizer is the algorithm that updates your model's weights to make predictions better. It's the "learning" part of machine learning.

**The Core Idea:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE LEARNING LOOP                                │
│                                                                     │
│  1. Model makes predictions                                         │
│                    │                                                │
│                    ▼                                                │
│  2. Loss function measures how wrong the predictions are            │
│                    │                                                │
│                    ▼                                                │
│  3. Backpropagation calculates gradients                            │
│     (which direction should each weight move?)                      │
│                    │                                                │
│                    ▼                                                │
│  4. OPTIMIZER updates weights based on gradients ◄── WE ARE HERE   │
│                    │                                                │
│                    ▼                                                │
│  5. Repeat from step 1                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Understanding `model.parameters()`

```python
model.parameters()
```

**What this returns:** An iterator (generator) over all trainable parameters in the model.

**What's a parameter?** A tensor of numbers that the model learns during training.

```python
# You can inspect them:
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# Output might look like:
# base_model.features.0.0.weight: torch.Size([32, 3, 3, 3]), requires_grad=False
# base_model.features.0.1.weight: torch.Size([32]), requires_grad=False
# ... (many frozen parameters)
# output_layer.weight: torch.Size([10, 1280]), requires_grad=True  ◄── TRAINABLE
# output_layer.bias: torch.Size([10]), requires_grad=True          ◄── TRAINABLE
```

**Key Point:** When we pass `model.parameters()` to the optimizer, it only updates parameters where `requires_grad=True`. The frozen MobileNetV2 parameters are passed but ignored.

### Understanding the Learning Rate (`lr=0.01`)

The learning rate controls how big each update step is.

```
                    Small lr (0.0001)          Large lr (0.1)
                    ─────────────────          ───────────────
                    
Gradient says       Take tiny steps            Take huge steps
"go this way" →     • • • • • • → goal         • ─ ─ → goal
                                                      ↑
                                                  Might overshoot!

Pros:               • Precise                  • Fast initially
                    • Stable                   
                    
Cons:               • Very slow                • Unstable
                    • Might get stuck          • Might never converge
```

**Visual analogy - Finding the lowest point in a valley:**

```
Learning Rate = How big your steps are when walking downhill blindfolded

Too small (lr=0.0001):          Just right (lr=0.01):           Too large (lr=0.5):
                                
    \_                              \_                              \_
      \_                              \_                              \_
        \•••••••••                      \_                              \     •
         \_      (takes forever)          •_                             \_  /
           \_                               •_                             •• 
             \___•                            •___•                      \___/
                  (goal)                          (goal)                     (bouncing!)
```

**Common learning rate values:**

| Learning Rate | When to Use |
|---------------|-------------|
| 0.1 - 0.01 | Starting point for SGD |
| 0.001 - 0.0001 | Common for Adam |
| 0.01 | Used here (relatively high for Adam) |

**Note:** The code uses `lr=0.01` which is relatively high for Adam. You might need to lower it if training is unstable.

### What is Adam?

Adam (Adaptive Moment Estimation) is one of the most popular optimizers. It combines ideas from two other optimizers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ADAM                                      │
│                                                                     │
│         Momentum                    +            RMSprop            │
│  (remember past gradients)              (adapt to each parameter)   │
│                                                                     │
│  "If I've been going                 "Parameters that get big       │
│   this direction, keep               gradients should take          │
│   going (with momentum)"             smaller steps"                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Adam's superpowers:**

| Feature | Benefit |
|---------|---------|
| Momentum | Helps push through small bumps and noise |
| Adaptive learning rates | Each parameter gets its own effective learning rate |
| Bias correction | Works well from the very first step |

**Adam vs other optimizers:**

```
SGD (Stochastic Gradient Descent):
• Simple: new_weight = old_weight - lr × gradient
• Can be effective but requires careful tuning
• Often needs learning rate schedules

Adam:
• Maintains running averages of gradients AND squared gradients
• Automatically adapts step size for each parameter
• Generally works well "out of the box"
• The most popular choice for deep learning
```

**The Adam Update Rule (simplified):**

```
For each parameter:
1. Update momentum:     m = 0.9 × m + 0.1 × gradient
2. Update velocity:     v = 0.999 × v + 0.001 × gradient²
3. Update weight:       weight = weight - lr × m / (√v + ε)

Where:
• m tracks the direction (momentum)
• v tracks the magnitude (for adaptive scaling)
• ε is a tiny number to prevent division by zero (default: 1e-8)
```

**You don't need to understand the math** - just know that Adam is smart about choosing step sizes!

### Complete Optimizer Breakdown

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

| Component | What It Is | What It Does |
|-----------|------------|--------------|
| `optimizer` | Variable name | Stores the optimizer object |
| `optim.Adam` | Optimizer class | Adam algorithm implementation |
| `model.parameters()` | Parameter iterator | Tells optimizer which weights to update |
| `lr=0.01` | Learning rate | Controls step size for updates |

**Additional Adam parameters (using defaults here):**

```python
# Full signature with defaults:
optim.Adam(
    params,              # Required: parameters to optimize
    lr=0.001,            # Learning rate (we're using 0.01)
    betas=(0.9, 0.999),  # Coefficients for momentum & velocity
    eps=1e-8,            # Small constant for numerical stability
    weight_decay=0,      # L2 regularization (0 = none)
    amsgrad=False        # Whether to use AMSGrad variant
)
```

---

## Deep Dive: The Loss Function (Criterion)

```python
criterion = nn.CrossEntropyLoss()
```

### What is a Loss Function?

A loss function (also called criterion or cost function) measures how wrong your model's predictions are. **Lower loss = better predictions.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOSS FUNCTION                               │
│                                                                     │
│    Model's                      True                                │
│    Prediction ──────►  LOSS  ◄────── Answer                         │
│                       FUNCTION                                      │
│                          │                                          │
│                          ▼                                          │
│                    Single number                                    │
│                  (how wrong were we?)                               │
│                                                                     │
│   Examples:                                                         │
│   • Predicted "shirt" when it was "shirt"     → Low loss (good!)   │
│   • Predicted "shirt" when it was "pants"     → High loss (bad!)   │
│   • Predicted "maybe shirt?" when it was "shirt" → Medium loss     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why CrossEntropyLoss?

**CrossEntropyLoss is THE standard loss function for classification tasks.**

It's designed specifically for problems where you're choosing between multiple categories (like our 10 clothing types).

### Understanding CrossEntropyLoss Step by Step

CrossEntropyLoss combines two operations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS ENTROPY LOSS                               │
│                                                                     │
│                         ┌─────────────┐                             │
│    Raw scores ────────► │   Softmax   │ ─────► Probabilities        │
│    (logits)             └─────────────┘        (sum to 1.0)         │
│                                                      │              │
│                                                      ▼              │
│                                              ┌─────────────┐        │
│    True label ─────────────────────────────► │ Neg Log     │ ───► Loss
│                                              │ Likelihood  │        │
│                                              └─────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Step 1: Softmax - Convert raw scores to probabilities**

```
Raw model output (logits):        After Softmax:
┌────────────────────────┐       ┌────────────────────────┐
│ Class 0 (t-shirt): 2.0 │       │ Class 0: 0.24  (24%)   │
│ Class 1 (pants):   1.0 │       │ Class 1: 0.09  (9%)    │
│ Class 2 (dress):   3.5 │  ───► │ Class 2: 0.54  (54%)   │ ◄── Highest!
│ Class 3 (coat):    0.5 │       │ Class 3: 0.05  (5%)    │
│ ...                    │       │ ...                    │
└────────────────────────┘       └────────────────────────┘
                                  Total = 1.00 (100%)

Softmax formula: probability_i = e^(score_i) / sum(e^(all_scores))
```

**Step 2: Negative Log Likelihood - Calculate the loss**

```
If the TRUE label is Class 2 (dress):

Loss = -log(probability of true class)
Loss = -log(0.54)
Loss = 0.62

┌─────────────────────────────────────────────────────────────┐
│              How -log(probability) works:                   │
│                                                             │
│  Probability │  -log(prob)  │  Interpretation              │
│  ────────────┼──────────────┼─────────────────────────────  │
│     1.0      │     0.0      │  Perfect! (100% confident    │
│              │              │  AND correct)                │
│     0.9      │     0.11     │  Very good                   │
│     0.5      │     0.69     │  Uncertain                   │
│     0.1      │     2.30     │  Wrong and confident = BAD   │
│     0.01     │     4.61     │  Very wrong = HIGH PENALTY   │
│                                                             │
│  Key insight: Being confidently wrong is heavily penalized! │
└─────────────────────────────────────────────────────────────┘
```

### Why is CrossEntropyLoss Good for Classification?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS ENTROPY LOSS BENEFITS                      │
│                                                                     │
│  1. ENCOURAGES CONFIDENCE                                           │
│     • Pushes model to give high probability to correct class        │
│     • Low loss only when model is confident AND correct             │
│                                                                     │
│  2. HEAVILY PENALIZES MISTAKES                                      │
│     • Being 99% sure of the wrong answer = very high loss           │
│     • This is good! We want the model to learn from big mistakes    │
│                                                                     │
│  3. MATHEMATICALLY NICE                                             │
│     • Gradients are well-behaved (not too big, not too small)       │
│     • Works well with softmax (they're designed together)           │
│                                                                     │
│  4. HANDLES MULTIPLE CLASSES                                        │
│     • Works for 2 classes or 1000 classes                           │
│     • The standard for image classification                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### CrossEntropyLoss Input/Output

```python
criterion = nn.CrossEntropyLoss()

# Inputs:
predictions = model(images)  # Shape: [batch_size, num_classes]
                             # e.g., [32, 10] - raw scores, NOT probabilities!

labels = batch_labels        # Shape: [batch_size]
                             # e.g., [32] - integers from 0 to 9

# Calculate loss:
loss = criterion(predictions, labels)  # Returns single scalar tensor

# Example:
# predictions[0] = [2.1, -0.5, 3.2, 0.1, ...]  (raw scores for image 0)
# labels[0] = 2                                 (true class is index 2)
# Loss computes how well 3.2 (class 2's score) dominates the others
```

**Important:** Pass RAW SCORES (logits), not softmax probabilities! CrossEntropyLoss applies softmax internally.

### Other Loss Functions (For Reference)

| Loss Function | Use Case |
|---------------|----------|
| `nn.CrossEntropyLoss()` | Multi-class classification (what we're using) |
| `nn.BCELoss()` | Binary classification (2 classes) |
| `nn.MSELoss()` | Regression (predicting continuous values) |
| `nn.L1Loss()` | Regression (more robust to outliers) |
| `nn.NLLLoss()` | Multi-class, but expects log-probabilities as input |

---

## How These Pieces Work Together in Training

```python
# The complete training loop (for context):

for epoch in range(num_epochs):
    for images, labels in train_loader:
        
        # Move data to same device as model
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Forward pass - model makes predictions
        predictions = model(images)
        
        # 2. Calculate loss - how wrong were we?
        loss = criterion(predictions, labels)
        
        # 3. Backward pass - calculate gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Calculate new gradients
        
        # 4. Update weights - optimizer adjusts parameters
        optimizer.step()       # Apply the gradients
```

**Visual flow:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING ITERATION                            │
│                                                                      │
│  Images ────► MODEL ────► Predictions                                │
│               (on device)      │                                     │
│                                ▼                                     │
│                          ┌──────────┐                                │
│  Labels ────────────────►│CRITERION │────► Loss (single number)      │
│                          │(CrossEnt)│           │                    │
│                          └──────────┘           │                    │
│                                                 ▼                    │
│                                          loss.backward()             │
│                                                 │                    │
│                                    Gradients calculated for          │
│                                    all trainable parameters          │
│                                                 │                    │
│                                                 ▼                    │
│                                         ┌────────────┐               │
│                                         │ OPTIMIZER  │               │
│                                         │  (Adam)    │               │
│                                         └────────────┘               │
│                                                 │                    │
│                                                 ▼                    │
│                                    Weights updated!                  │
│                                    Model is now slightly better      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Complete Setup Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING SETUP                                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ DEVICE                                                      │   │
│  │ • Determines where computations happen (CPU vs GPU)         │   │
│  │ • GPU = much faster training                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ MODEL                                                       │   │
│  │ • Contains all the neural network layers                    │   │
│  │ • Holds trainable parameters (weights & biases)             │   │
│  │ • Must be on same device as data                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│              ┌───────────────┴───────────────┐                      │
│              ▼                               ▼                      │
│  ┌────────────────────────┐    ┌────────────────────────────┐      │
│  │ OPTIMIZER              │    │ CRITERION                  │      │
│  │ • Algorithm for        │    │ • Measures prediction      │      │
│  │   updating weights     │    │   error                    │      │
│  │ • Adam = adaptive,     │    │ • CrossEntropyLoss =       │      │
│  │   momentum-based       │    │   standard for             │      │
│  │ • lr controls step     │    │   classification           │      │
│  │   size                 │    │ • Lower = better           │      │
│  └────────────────────────┘    └────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Common Beginner Questions

### Q: Why do we need `model.to(device)` AND `images.to(device)` later?
Both the model AND the data must be on the same device. The model is moved once during setup. Data must be moved each batch during training.

### Q: What happens if I forget `optimizer.zero_grad()`?
Gradients accumulate! Your updates will be based on the sum of ALL previous gradients, not just the current batch. This usually breaks training.

### Q: Is Adam always the best optimizer?
No, but it's a great default. Some situations where others might be better:

| Scenario | Consider |
|----------|----------|
| Large-scale image models | SGD with momentum often generalizes better |
| Transformer models | AdamW (Adam with proper weight decay) |
| Limited memory | SGD uses less memory than Adam |

### Q: How do I know if my learning rate is good?
Watch the loss:
- **Decreasing steadily** → Good learning rate
- **Decreasing very slowly** → Try increasing lr
- **Jumping around wildly** → Try decreasing lr
- **Increasing** → lr is way too high

### Q: What if I have multiple GPUs?
You can use `nn.DataParallel` or `nn.DistributedDataParallel` to train across multiple GPUs. That's an advanced topic!

### Q: Why does CrossEntropyLoss expect raw scores, not probabilities?
Computing log-softmax directly from raw scores is more numerically stable than computing softmax first, then log. PyTorch combines them for better precision.

---

## Quick Reference

| Variable | Type | Purpose |
|----------|------|---------|
| `device` | `torch.device` | Where to run computations (CPU/GPU) |
| `model` | `nn.Module` subclass | The neural network with all its layers |
| `optimizer` | `optim.Optimizer` | Updates weights using gradients |
| `criterion` | `nn.Module` (loss) | Measures prediction error |

| Hyperparameter | Our Value | What It Controls |
|----------------|-----------|------------------|
| `num_classes` | 10 | Output size (clothing categories) |
| `lr` | 0.01 | Step size for weight updates |

---

## Summary

| Concept | Key Points |
|---------|------------|
| **Device** | GPU (cuda) is much faster than CPU; model and data must be on same device |
| **Model** | Contains layers, parameters, and defines forward pass; lives on a device |
| **Optimizer** | Updates weights to minimize loss; Adam is adaptive and popular; lr controls step size |
| **Criterion** | Measures how wrong predictions are; CrossEntropyLoss is standard for classification |


# Training Loop (Step 6)

# The PyTorch Training Loop: A Beginner's Complete Guide

This guide provides a detailed breakdown of a complete PyTorch training loop. We'll cover every line of code, explain why each step is necessary, and dive deep into concepts like gradient zeroing and train/eval modes.

*Note: This code is from the MLZoomcamp course.*

---

## The Full Code

```python
# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the training data
    for inputs, labels in train_loader:
        # Move data to the specified device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients to prevent accumulation
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        running_loss += loss.item()
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        # Update total and correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Disable gradient calculation for validation
    with torch.no_grad():
        # Iterate over the validation data
        for inputs, labels in val_loader:
            # Move data to the specified device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)

            # Accumulate validation loss
            val_loss += loss.item()
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            # Update total and correct predictions
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # Print epoch results
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
```

---

## Understanding PyTorch: A Lower-Level Framework

Before diving in, it's important to understand that **PyTorch is a lower-level framework**. Unlike higher-level frameworks (like Keras), PyTorch doesn't automatically handle things like:

- Calculating accuracy
- Switching between training and evaluation modes
- Zeroing gradients
- Moving data between devices

**This is by design.** PyTorch gives you full control and visibility into what's happening, which is why we need to implement these things ourselves. This makes PyTorch more flexible and educational, but also means more code.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL vs LOW-LEVEL                          │
│                                                                     │
│  High-Level (Keras):              Low-Level (PyTorch):              │
│  ─────────────────                ────────────────────              │
│  model.fit(X, y, epochs=10)       for epoch in range(10):           │
│                                       for batch in loader:          │
│  (One line does everything!)              zero_grad()               │
│                                           forward()                 │
│                                           loss()                    │
│                                           backward()                │
│                                           step()                    │
│                                           calculate_metrics()       │
│                                                                     │
│  Pros: Simple, fast to write      Pros: Full control, transparent   │
│  Cons: Less control, "magic"      Cons: More code, more to learn    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Big Picture: Training Loop Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPLETE TRAINING STRUCTURE                    │
│                                                                     │
│  for each EPOCH (full pass through entire dataset):                 │
│  │                                                                  │
│  ├──► TRAINING PHASE                                                │
│  │    │                                                             │
│  │    ├── Set model.train() mode                                    │
│  │    │                                                             │
│  │    └── for each BATCH:                                           │
│  │        ├── Move data to device                                   │
│  │        ├── Zero gradients                                        │
│  │        ├── Forward pass (predictions)                            │
│  │        ├── Calculate loss                                        │
│  │        ├── Backward pass (gradients)                             │
│  │        ├── Update weights                                        │
│  │        └── Track metrics                                         │
│  │                                                                  │
│  ├──► VALIDATION PHASE                                              │
│  │    │                                                             │
│  │    ├── Set model.eval() mode                                     │
│  │    │                                                             │
│  │    └── with torch.no_grad():                                     │
│  │        └── for each BATCH:                                       │
│  │            ├── Move data to device                               │
│  │            ├── Forward pass (predictions)                        │
│  │            ├── Calculate loss                                    │
│  │            └── Track metrics                                     │
│  │                                                                  │
│  └──► PRINT RESULTS                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Line-by-Line Explanation

### Lines 1-2: Setting Up the Loop

```python
# Training loop
num_epochs = 10

for epoch in range(num_epochs):
```

#### `num_epochs = 10`

**What this does:** Sets how many complete passes through the training data we'll make.

**Key Concept - What is an Epoch?**

```
One EPOCH = One complete pass through ALL training data

If you have 1000 training images and batch_size=32:
├── Batch 1:   images 1-32
├── Batch 2:   images 33-64
├── Batch 3:   images 65-96
├── ...
└── Batch 32:  images 993-1000 (last batch is smaller)

After all 32 batches → 1 epoch complete!

With num_epochs=10, we do this 10 times.
Each time, the model sees every image again (in different order if shuffle=True)
```

**How many epochs should you use?**

| Too Few Epochs | Just Right | Too Many Epochs |
|----------------|------------|-----------------|
| Model hasn't learned enough | Model generalizes well | Model memorizes training data |
| Low train & val accuracy | Good train & val accuracy | High train acc, low val acc |
| Underfitting | Sweet spot | Overfitting |

**Common practice:** Train until validation accuracy stops improving (early stopping).

#### `for epoch in range(num_epochs):`

**What this does:** Creates a loop that runs 10 times (epochs 0 through 9).

Each iteration of this outer loop represents one complete epoch of training.

---

## The Training Phase

### Lines 5-9: Training Phase Setup

```python
    # Training phase
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
```

#### `model.train()` - CRITICAL CONCEPT

**What this does:** Sets the model to training mode.

**Why is this necessary?** Certain layers behave DIFFERENTLY during training vs evaluation:

```
┌────────────────────────────────────────────────────────────────────┐
│                    model.train() vs model.eval()                   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ DROPOUT LAYERS                                              │   │
│  │                                                             │   │
│  │ model.train():                  model.eval():               │   │
│  │ ┌───┬───┬───┬───┬───┐          ┌───┬───┬───┬───┬───┐        │   │
│  │ │ ● │ ○ │ ● │ ○ │ ● │          │ ● │ ● │ ● │ ● │ ● │        │   │
│  │ └───┴───┴───┴───┴───┘          └───┴───┴───┴───┴───┘        │   │
│  │ ● = active   ○ = dropped        All neurons active          │   │
│  │ (random ~50% dropped)           (no randomness)             │   │
│  │                                                             │   │
│  │ Purpose: Prevents overfitting   Purpose: Use full model     │   │
│  │          during training                 for predictions    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ BATCHNORM LAYERS                                            │   │
│  │                                                             │   │
│  │ model.train():                  model.eval():               │   │
│  │ • Calculates mean & variance    • Uses STORED running       │   │
│  │   from CURRENT BATCH              mean & variance           │   │
│  │ • Updates running statistics    • No updates to statistics  │   │
│  │                                                             │   │
│  │ Purpose: Learn normalization    Purpose: Consistent         │   │
│  │          parameters                      predictions        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  IMPORTANT: Always call model.train() before training and          │
│             model.eval() before validation/inference!              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**What happens if you forget?**

| Forgot `model.train()` | Forgot `model.eval()` |
|------------------------|----------------------|
| Dropout doesn't drop neurons | Dropout randomly drops during evaluation |
| BatchNorm uses old statistics | BatchNorm stats vary with each batch |
| Model may not learn properly | Inconsistent, random predictions |

#### Tracking Variables

```python
running_loss = 0.0  # Accumulates loss across all batches
correct = 0         # Counts correct predictions
total = 0           # Counts total samples seen
```

These variables track metrics across the entire epoch. They're reset to zero at the start of each epoch.

---

### Lines 11-14: Iterating Through Batches

```python
    # Iterate over the training data
    for inputs, labels in train_loader:
        # Move data to the specified device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)
```

#### `for inputs, labels in train_loader:`

**What this does:** Loops through the training data, one batch at a time.

**What you get each iteration:**

```
train_loader yields tuples of (inputs, labels)

inputs: Tensor of shape [batch_size, channels, height, width]
        Example: [32, 3, 224, 224]
        (32 images, 3 color channels, 224x224 pixels)

labels: Tensor of shape [batch_size]
        Example: [32]
        (32 integer labels, one per image)
        Values: 0, 1, 2, ... (up to num_classes - 1)
```

#### `inputs, labels = inputs.to(device), labels.to(device)`

**What this does:** Moves both tensors to the same device as the model (GPU or CPU).

**This is done EVERY batch** because data comes from the DataLoader on CPU by default.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA MOVEMENT                                │
│                                                                     │
│    DataLoader                                                       │
│  (loads from disk)                                                  │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────┐     .to(device)      ┌──────────┐                     │
│  │  inputs  │ ──────────────────►  │  inputs  │                     │
│  │  (CPU)   │                      │  (GPU)   │                     │
│  └──────────┘                      └──────────┘                     │
│                                          │                          │
│                                          ▼                          │
│                                    ┌──────────┐                     │
│                                    │  MODEL   │                     │
│                                    │  (GPU)   │                     │
│                                    └──────────┘                     │
│                                                                     │
│  Both must be on the same device for computations to work!          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Lines 16-17: Zero Gradients - CRITICAL STEP

```python
        # Zero the parameter gradients to prevent accumulation
        optimizer.zero_grad()
```

**What this does:** Clears all gradient values from the previous batch.

**Why is this critical?**

In PyTorch, **gradients are accumulated by default**. This means if you don't zero the gradients before calculating the gradients for the current batch, the gradients from the previous batch will be added to the gradients of the current batch. This would lead to incorrect updates to your model's parameters.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GRADIENT ACCUMULATION PROBLEM                    │
│                                                                     │
│  WITHOUT zero_grad():                                               │
│  ────────────────────                                               │
│                                                                     │
│  Batch 1: gradient = 0.5                                            │
│           (stored in param.grad)                                    │
│                     │                                               │
│                     ▼                                               │
│  Batch 2: gradient calculated = 0.3                                 │
│           param.grad = 0.5 + 0.3 = 0.8  ◄── WRONG!                  │
│           (accumulated, not replaced)                               │
│                     │                                               │
│                     ▼                                               │
│  Batch 3: gradient calculated = -0.2                                │
│           param.grad = 0.8 + (-0.2) = 0.6  ◄── EVEN MORE WRONG!     │
│                                                                     │
│  Result: Updates are based on sum of ALL past gradients!            │
│          Model learns garbage.                                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WITH zero_grad():                                                  │
│  ─────────────────                                                  │
│                                                                     │
│  Batch 1: zero_grad() → param.grad = 0                              │
│           gradient calculated = 0.5                                 │
│           param.grad = 0.5  ◄── Correct!                            │
│                     │                                               │
│                     ▼                                               │
│  Batch 2: zero_grad() → param.grad = 0                              │
│           gradient calculated = 0.3                                 │
│           param.grad = 0.3  ◄── Correct!                            │
│                     │                                               │
│                     ▼                                               │
│  Batch 3: zero_grad() → param.grad = 0                              │
│           gradient calculated = -0.2                                │
│           param.grad = -0.2  ◄── Correct!                           │
│                                                                     │
│  Result: Each update is based only on current batch.                │
│          Model learns properly!                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**When should you NOT zero gradients?**

There's actually one case where accumulating gradients is useful: **gradient accumulation for large effective batch sizes**. If you can only fit batch_size=8 in memory but want the effect of batch_size=32:

```python
# Advanced technique: accumulate for 4 batches before updating
for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / 4  # Scale loss
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % 4 == 0:  # Every 4 batches
        optimizer.step()  # Update with accumulated gradients
        optimizer.zero_grad()  # Now zero for next accumulation
```

But for normal training, **always zero gradients before backward()**.

---

### Lines 18-19: Forward Pass

```python
        # Forward pass
        outputs = model(inputs)
```

**What this does:** Passes the input images through the model to get predictions.

**What happens inside:**

```
inputs: [32, 3, 224, 224]
        (32 images)
            │
            ▼
    ┌───────────────┐
    │     MODEL     │
    │               │
    │ MobileNetV2   │──► Feature extraction
    │ features      │
    │               │
    │ AvgPool       │──► Reduce to 1x1
    │               │
    │ Flatten       │──► [32, 1280]
    │               │
    │ Linear        │──► Classification
    │               │
    └───────────────┘
            │
            ▼
outputs: [32, 10]
         (32 sets of 10 class scores)

Example outputs[0]:
[2.1, -0.5, 3.8, 0.2, -1.0, 0.8, 1.2, -0.3, 0.1, 0.4]
 │                │
 └── Class 0      └── Class 2 has highest score (prediction)
     score
```

---

### Lines 20-21: Calculate Loss

```python
        # Calculate the loss
        loss = criterion(outputs, labels)
```

**What this does:** Measures how wrong the predictions are.

```
outputs: [32, 10] ──────┐
(raw scores)            │
                        ▼
                 ┌─────────────────┐
                 │ CrossEntropyLoss│
                 └─────────────────┘
                        ▲
labels: [32] ───────────┘
(true classes, e.g., [2, 0, 5, 2, ...])

                        │
                        ▼
                 
loss: single scalar tensor
      e.g., tensor(1.8234, grad_fn=<NllLossBackward>)

Note: The "grad_fn" indicates this tensor is connected to
      the computation graph for backpropagation!
```

---

### Lines 22-24: Backward Pass and Optimization

```python
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
```

#### `loss.backward()` - Backpropagation

**What this does:** Calculates gradients for ALL trainable parameters.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKPROPAGATION                               │
│                                                                     │
│  Forward pass built a computation graph:                            │
│                                                                     │
│  inputs → conv1 → relu → conv2 → ... → linear → outputs → loss      │
│                                                                     │
│  loss.backward() walks BACKWARD through this graph:                 │
│                                                                     │
│  loss → outputs → linear → ... → conv2 → relu → conv1               │
│    │                  │                            │                │
│    ▼                  ▼                            ▼                │
│  ∂loss/∂loss    ∂loss/∂weights            ∂loss/∂weights            │
│    = 1          of linear layer           of conv1                  │
│                                                                     │
│  For each parameter, calculates:                                    │
│  "How much would the loss change if I tweaked this weight?"         │
│                                                                     │
│  These gradients are STORED in param.grad for each parameter        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key point:** This only CALCULATES gradients. It doesn't change any weights yet!

#### `optimizer.step()` - Update Weights

**What this does:** Uses the calculated gradients to update all parameters.

```
For each trainable parameter:

new_weight = old_weight - learning_rate × gradient

(Adam is more sophisticated, but this is the basic idea)

Example:
┌─────────────────────────────────────────────────────────────────────┐
│  Before:    weight = 0.5,  gradient = 0.1,  lr = 0.01               │
│  After:     weight = 0.5 - (0.01 × 0.1) = 0.499                     │
│                                                                     │
│  The weight moved slightly in the direction that reduces loss!      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Lines 26-31: Tracking Metrics

```python
        # Accumulate training loss
        running_loss += loss.item()
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        # Update total and correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
```

#### `running_loss += loss.item()`

**What this does:** Adds this batch's loss to the running total.

| Method | Returns | Use Case |
|--------|---------|----------|
| `loss` | Tensor with gradients | For backpropagation |
| `loss.item()` | Python float | For logging/tracking (no gradient history) |

**Why `.item()`?** Using the raw tensor would accumulate computation graph history, eating up memory. `.item()` extracts just the number.

#### `_, predicted = torch.max(outputs.data, 1)`

**What this does:** Gets the predicted class for each image.

```python
torch.max(outputs.data, 1)  # Returns (max_values, max_indices)

Example:
outputs = [[2.1, -0.5, 3.8, ...],   # Image 0: class 2 wins
           [4.2,  0.1, 1.0, ...],   # Image 1: class 0 wins
           ...]

max_values = [3.8, 4.2, ...]  # The highest scores (we ignore these with _)
max_indices = [2, 0, ...]     # The winning class indices (this is 'predicted')
```

**Breaking it down:**

| Part | Meaning |
|------|---------|
| `outputs.data` | The tensor data (without gradient tracking) |
| `1` | Find max along dimension 1 (the class dimension) |
| `_` | Python convention: "I don't need this value" |
| `predicted` | Tensor of predicted class indices |

#### `total += labels.size(0)`

**What this does:** Adds the batch size to the running count.

```python
labels.size(0)  # Returns batch_size (e.g., 32)
                # Dimension 0 is the batch dimension
```

#### `correct += (predicted == labels).sum().item()`

**What this does:** Counts how many predictions were correct.

```python
# Step by step:
predicted == labels
# Returns: tensor([True, False, True, True, False, ...])
#          (Boolean for each image)

(predicted == labels).sum()
# Returns: tensor(24)  (count of True values)

(predicted == labels).sum().item()
# Returns: 24  (as Python int)
```

---

### Lines 33-35: Calculate Epoch Metrics

```python
    # Calculate average training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
```

#### `train_loss = running_loss / len(train_loader)`

**What this does:** Calculates the average loss per batch.

```
running_loss = sum of all batch losses (e.g., 45.6)
len(train_loader) = number of batches (e.g., 32)

train_loss = 45.6 / 32 = 1.425
```

#### `train_acc = correct / total`

**What this does:** Calculates the accuracy (proportion correct).

```
correct = number of correct predictions (e.g., 856)
total = total predictions made (e.g., 1000)

train_acc = 856 / 1000 = 0.856 (85.6% accuracy)
```

---

## The Validation Phase

### Lines 37-41: Validation Setup

```python
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
```

#### `model.eval()` - CRITICAL CONCEPT

**What this does:** Sets the model to evaluation mode.

**Why is this necessary?** As explained earlier, `model.eval()` changes how certain layers behave:

- **Dropout layers** become inactive (pass through all neurons instead of randomly dropping)
- **BatchNorm layers** use their accumulated running statistics instead of calculating from the current batch

This ensures consistent, reproducible behavior during inference and prevents randomness from affecting the evaluation results.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHY model.eval() MATTERS                         │
│                                                                     │
│  Imagine evaluating the same image twice in training mode:          │
│                                                                     │
│  model.train()                                                      │
│  prediction_1 = model(image)  →  "pants" (70% confident)            │
│  prediction_2 = model(image)  →  "shirt" (60% confident)  ← RANDOM! │
│                                                                     │
│  Dropout randomly disabled different neurons each time!             │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  model.eval()                                                       │
│  prediction_1 = model(image)  →  "pants" (85% confident)            │
│  prediction_2 = model(image)  →  "pants" (85% confident)  ← SAME!   │
│                                                                     │
│  Consistent predictions for evaluation!                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Lines 43-44: Disable Gradient Calculation

```python
    # Disable gradient calculation for validation
    with torch.no_grad():
```

**What this does:** Disables gradient tracking for all operations inside this block.

**Why do this?**

| With Gradients | Without Gradients (`torch.no_grad()`) |
|----------------|--------------------------------------|
| Builds computation graph | No computation graph |
| Stores intermediate values | Frees memory immediately |
| Needed for training | Perfect for inference |
| Slower, uses more memory | Faster, uses less memory |

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GRADIENT TRACKING OVERHEAD                       │
│                                                                     │
│  Normal forward pass:                                               │
│  ────────────────────                                               │
│  input → [save for grad] → conv → [save for grad] → relu → ...      │
│                                                                     │
│  Memory: High (storing intermediate values for backward pass)       │
│  Speed:  Slower (extra bookkeeping)                                 │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  with torch.no_grad():                                              │
│  ─────────────────────                                              │
│  input → conv → relu → ...  (just compute, don't save)              │
│                                                                     │
│  Memory: Low (no storage for gradients)                             │
│  Speed:  Faster (no bookkeeping)                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**During validation we don't need gradients because:**
1. We're not updating weights
2. We just want predictions
3. It would waste memory and time

---

### Lines 45-58: Validation Loop

```python
        # Iterate over the validation data
        for inputs, labels in val_loader:
            # Move data to the specified device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)

            # Accumulate validation loss
            val_loss += loss.item()
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            # Update total and correct predictions
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
```

**This is similar to training, but notice what's MISSING:**

```
Training loop:                     Validation loop:
──────────────                     ────────────────
inputs, labels = ...               inputs, labels = ...
inputs, labels = inputs.to(...)    inputs, labels = inputs.to(...)

optimizer.zero_grad()  ◄── MISSING (no gradients to zero)

outputs = model(inputs)            outputs = model(inputs)
loss = criterion(...)              loss = criterion(...)

loss.backward()        ◄── MISSING (no backprop needed)
optimizer.step()       ◄── MISSING (no weight updates)

running_loss += ...                val_loss += ...
_, predicted = ...                 _, predicted = ...
total += ...                       val_total += ...
correct += ...                     val_correct += ...
```

**Validation is "forward pass only"** - we just measure how well the model performs without changing it.

---

### Lines 60-62: Calculate Validation Metrics

```python
    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
```

Same calculation as training metrics, but for the validation set.

---

### Lines 64-67: Print Results

```python
    # Print epoch results
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
```

**What this outputs:**

```
Epoch 1/10
  Train Loss: 1.8234, Train Acc: 0.4532
  Val Loss: 1.5621, Val Acc: 0.5123

Epoch 2/10
  Train Loss: 1.2456, Train Acc: 0.6234
  Val Loss: 1.1234, Val Acc: 0.6543

...
```

**How to interpret these numbers:**

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Train Loss | Decreasing | Staying high or increasing |
| Train Acc | Increasing | Staying low |
| Val Loss | Decreasing | Increasing (overfitting!) |
| Val Acc | Increasing | Decreasing (overfitting!) |

**Healthy training looks like:**

```
Epoch  │ Train Loss │ Train Acc │ Val Loss │ Val Acc
───────┼────────────┼───────────┼──────────┼─────────
  1    │   2.30     │   0.35    │   1.80   │   0.45   
  2    │   1.50     │   0.55    │   1.20   │   0.60   
  3    │   0.90     │   0.70    │   0.85   │   0.72   
  4    │   0.60     │   0.82    │   0.65   │   0.80   
  5    │   0.40     │   0.88    │   0.55   │   0.84   
                        ↑             ↑
                   Both improving together = good!
```

**Overfitting looks like:**

```
Epoch  │ Train Loss │ Train Acc │ Val Loss │ Val Acc
───────┼────────────┼───────────┼──────────┼─────────
  1    │   2.30     │   0.35    │   1.80   │   0.45   
  5    │   0.40     │   0.88    │   0.55   │   0.84   
 10    │   0.05     │   0.99    │   0.90   │   0.80   ← Val getting worse!
 15    │   0.01     │   1.00    │   1.50   │   0.75   ← Model memorized training data
                        ↑             ↑
                 Train perfect, but val suffering = overfitting!
```

---

## Complete Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                     COMPLETE TRAINING EPOCH                        │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    TRAINING PHASE                           │   │
│  │                                                             │   │
│  │  model.train() ─────────────────────────────────────────►   │   │
│  │                                                             │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │  FOR EACH BATCH:                                      │  │   │
│  │  │                                                       │  │   │
│  │  │  [Data to GPU] → [Zero Grad] → [Forward] → [Loss]     │  │   │
│  │  │                                    │                  │  │   │
│  │  │                                    ▼                  │  │   │
│  │  │                           [Backward] → [Step]         │  │   │
│  │  │                                    │                  │  │   │
│  │  │                                    ▼                  │  │   │
│  │  │                           [Track Metrics]             │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                             │   │
│  │  → Calculate train_loss, train_acc                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   VALIDATION PHASE                          │   │
│  │                                                             │   │
│  │  model.eval() ──────────────────────────────────────────►   │   │
│  │                                                             │   │
│  │  with torch.no_grad(): ─────────────────────────────────►   │   │
│  │                                                             │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │  FOR EACH BATCH:                                      │  │   │
│  │  │                                                       │  │   │
│  │  │  [Data to GPU] → [Forward] → [Loss] → [Track Metrics] │  │   │
│  │  │                                                       │  │   │
│  │  │  (No backward, no step - just measure performance!)   │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                             │   │
│  │  → Calculate val_loss, val_acc                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  PRINT RESULTS                                              │   │
│  │                                                             │   │
│  │  Epoch 1/10                                                 │   │
│  │    Train Loss: 1.8234, Train Acc: 0.4532                    │   │
│  │    Val Loss: 1.5621, Val Acc: 0.5123                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ═══════════════════════════════════════════════════════════════   │
│                          REPEAT FOR NEXT EPOCH                     │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Common Beginner Questions

### Q: Why do we need both `model.train()` and `model.eval()`?
These methods control layer behaviors that differ between training and inference. Dropout and BatchNorm act differently in each mode. Forgetting to switch modes is a common bug that causes mysterious performance issues.

### Q: What if I forget `optimizer.zero_grad()`?
Gradients accumulate across batches, leading to incorrect weight updates. Your model will learn poorly or not at all. This is one of the most common PyTorch bugs!

### Q: Why calculate loss during validation if we're not training?
To monitor for overfitting. If validation loss starts increasing while training loss decreases, your model is memorizing rather than learning.

### Q: Why use `.item()` instead of just the tensor value?
Using `.item()` extracts a Python number without gradient history. This prevents memory leaks from accumulating computation graph references.

### Q: Can I skip the validation phase?
Technically yes, but you'd be flying blind. Validation tells you if your model generalizes. Without it, you might train a model that only works on training data.

### Q: Why is batch size 32 so common?
It's a balance between:
- **Too small (1-8):** Noisy gradients, slow training
- **Just right (16-64):** Stable gradients, efficient GPU usage
- **Too large (256+):** May need larger learning rate, can hurt generalization

32 fits well on most GPUs and provides stable training.

---

## Quick Reference Card

| Step | Code | Purpose |
|------|------|---------|
| Training mode | `model.train()` | Enable dropout, update BatchNorm stats |
| Zero gradients | `optimizer.zero_grad()` | Clear old gradients |
| Forward pass | `outputs = model(inputs)` | Get predictions |
| Calculate loss | `loss = criterion(outputs, labels)` | Measure error |
| Backward pass | `loss.backward()` | Calculate gradients |
| Update weights | `optimizer.step()` | Apply gradients |
| Eval mode | `model.eval()` | Disable dropout, freeze BatchNorm |
| No gradients | `with torch.no_grad():` | Save memory during inference |

---

## Summary

| Concept | Key Points |
|---------|------------|
| **Epoch** | One complete pass through all training data |
| **model.train()** | Enables training behaviors (dropout active, BatchNorm updates) |
| **model.eval()** | Enables evaluation behaviors (dropout off, BatchNorm frozen) |
| **optimizer.zero_grad()** | MUST be called before backward() to prevent gradient accumulation |
| **loss.backward()** | Calculates gradients for all trainable parameters |
| **optimizer.step()** | Updates weights using the calculated gradients |
| **torch.no_grad()** | Disables gradient tracking for efficiency during inference |
| **Validation** | Measures generalization; watch for overfitting |