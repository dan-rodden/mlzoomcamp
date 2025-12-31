# 8.1 Deep Learning

- deep learning is especially helpful with image data and multi-classification.

### High Level Overview
1. Customer comes to website and uploads an image
2. An outside service called `Fashion Classification Service` replies with a a suggested category. A neural network will predict the category of the image and lives inside the `Fashion classification service`.
3. Goal is to make it simpler for a user to create a listing.

<img src="08-images/Screenshot 2025-12-04 at 7.53.58 PM.png">

# 8.2 Tensorflow and Keras
- tensorflow is a library for doing deep learning and framework while keras is a higher level abstraction on top of tensorflow. Keras makes it simpler to manipulate and train neural networks.

- one can use `from tensorflow.keras.preprocessing.image import load_img` to load images. One needs to use'target_size' argument in `load_img(fullname, target_size=(299, 299))` because NN's require specific size of images. 

- the library for processing images is called PIL (python image library).

- images are represented internally as an array with three channels. There is a red channel, blue channel, and green channel. In each channel, there is an array of values. This array corresponds to the image size. Each pixel in the array takes a values between 0-255 (1-byte) 

- the dimensions of the array will be height * width * number_of_channels. Therefore, (150, 150, 3) argument will be a 150 by 150 image with 3 channels.

<img src="08-images/Screenshot 2025-12-05 at 9.49.48 AM.png">

- keras has pre-trained neural networks (the NN's differ by their layer architecture). Just go to the website and choose a pre-trained model. Use the Xception model

- to grab the pre-trained Xception model that used imagenet for training data (means the model is pre-trained) use the following:
```python
from tensorflow.keras.applications.xception import Xception
model = Xception(weights='imagenet', input_shape=(299, 299, 3))
```

- when using the model to classify images, it expects multiple images as input for classification. Therefore, one needs to use a list structure to supply the images ie `X = np.array([x])`.

- one also must preprocess the images in the same manner that the model was trained. Use `from tensorflow.keras.applications.xception import preprocess_input` to make sure that the model can recognize the image being supplied.

- one thing to be careful of is if the annotations in the training set you are using has the correct category. When one supplies a t-shirt for classification to imagenet, the item is misclassified because there is no training data for the item. One needs to have the correct weights/data for proper classification.

- the whole thing put together
```python
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

model = Xception(weights='imagenet', input_shape=(299, 299, 3))

# use the above model to classify the image of a t-shirt
X = np.array([x])
X = preprocess_input(X)

pred = model.predict(X)

# (1 means there is one image used, and 1000 means there are 1000 possible classes for the image to be in)
# each value in pred is the probability that the image belongs to that class
pred.shape # (1, 1000)

# a function that maps indices of the array to class names
# it looks at the predictions and tries to make them human readable
# even though imagenet is very comprehensive, when it comes to clothes it doesn't work for our classification task
decode_predictions(pred)
```

# 8.4 Convolutional Neural Network

## How CNNs work

- CNNs are built from multiple layers. Each layer is built of filters. The filter looks at each pixel in the image and generates a number stating how similar the filter is to pixel. This creates an array/table of similarity on the image called a `feature map`. Higher values mean there is higher similarity.
- For example, if we have 1 layer neural network with 6 filters, then we will have a `feature map` for every filter. With 1 layer and 6 filters this would give 6 feature maps.

<img src="08-images/Screenshot 2025-12-05 at 12.43.10 PM.png">

- this shows the result of using three filters. This means we have one feature map per filter. A set of feature maps is the the output of a single layer in a CNN.

<img src="08-images/Screenshot 2025-12-05 at 12.45.42 PM.png">

- one can then use the output of the first layer (a feature map) as input into the second CNN layer which has another set of filters. This creates a chain of layers and feature maps.

<img src="08-images/Screenshot 2025-12-05 at 12.48.59 PM.png">

### This part is difficult

- the first layer of a CNN may start out with very simple features. The lesson shows two half circles of different orientations along with lines and slashes. The first CNN detects and learns from these shapes. The next CNN layer will further refine the broad categories identified in the image map. For instance, the two half circles could be combined into a feature that detects a circle. Two dashes of differing orientations can be combined to give a cross features (resembles the shape of X since / + \ = X)

- by the third layer you have even more complex and refined shapes. Using the filtering through the differing layers one can arrive at more refined high-level features.

<img src="08-images/Screenshot 2025-12-05 at 12.55.59 PM.png">

- the result of using the convolutional layers on the image is a vector that captures all of the feature information about the image. 

- from the vector we want to build a model from this vector to make predictions. If we did binary classification, 

### How one goes from binary classification with logistic regression to a neural network.

- with logistic regression we start out with a vector of weights. We multiply the individual values of the vector by the weights giving a linear combination. This linear combination is converted to a probability using the logit function/sigmoid.

<img src="08-images/Screenshot 2025-12-05 at 1.14.55 PM.png">

- if we want to build a model for multiple classes. The process is largely the same. However, we build three different models and, therefore, the three different models have different weights resulting in differing linear combinations for each model from the same vector. Finally, we use the `SOFTMAX` objective function to perform the classification. 

<img src="08-images/Screenshot 2025-12-05 at 1.18.44 PM.png">

- the concept of using multiple Logistic Regression models to classify a shirt is analogous to the dense layer of a neural network with the dense layer just being the three models bundled together. Instead of having three individual weight vectors (w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub>) the dense layer combines them into a single matrix W. One then just multiplies W*x giving the result. 

- the purpose of the convolutional layers (feature extraction) comes before the dense layer. The CNN layers transform, in this case, the image input of the t-shirt into a vector that can be understood by our machine learning method. In this case, the dense layer can be multiplied by the transformed representation of the t-shirt (its a vector) allowing for classification.

<img src="08-images/Screenshot 2025-12-05 at 1.57.26 PM.png">

- one then chain dense layers together

<img src="08-images/Screenshot 2025-12-05 at 1.59.49 PM.png">

This is a summary:
- One starts with a t-shirt
- One uses CNN layers (build of different filters) to extract features and turn the t-shirt into a vector format.
- This vector representation is then put through a variety of dense layers to arrive finally at a prediction.

<img src="08-images/Screenshot 2025-12-05 at 2.02.23 PM.png">

Useful Links
- https://cs231n.github.io/


# 8.5 Transfer Learning

- if one grabs an already trained model such as imagenet, there are two major components of the model. Namely, the convolutional layers, and the dense layers.
    - the convolutional layers - are generic, difficult to generate as one needs a large dataset of images, and computationally intensive. The convolutional layers output the vector representation of a photograph. This vector representation is in some sense universal with these features working with a variety of learning methods and approaches
    - the dense layers - are specific to imagenet. These come from the already predefined classes that imagenet uses. However, we may have a different set of classes or labels. This means that the __dense layers tied to imagenet are NOT useful__. However, we can still use the convolutional layers to get our image data into a proper format for making predictions.
 - In __transfer learning__ we use the convolutional layers to convert our batch of data into a vector format, but we throw away the dense layers as they will not allow for us to proper classify data in our batch. This knowledge from the convolutional layers is transferred to us.

<img src="08-images/Screenshot 2025-12-05 at 4.11.47 PM.png">

- Workflow to convert a batch of training data using the ImageNet's convolutional layers into vector representation. There are a few things to be aware of:
    - it is a good idea to keep the size of images small so allow for faster training/conversion thereby speeding up experimentation
    - if we have 32 pictures in our dataset with dimensions of 150 by 150 with 3 color channels this gives a (32, 150, 150, 3) sized object.
    - the output of this object will be 32 image vectors forming an image matrix X
    - the key arguments for running the `train_gen.flow_from_directory` are:
        1. the directory location: `../datasets/clothing-dataset-small-master/train`
        2. the target_size (image size): `target_size=(150, 150)`
        3. the training size (# of images in the train data): `batch_size=32`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator # import the function to preprocess the images

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input) # function to preprocess the images
# generate the training data
train_ds = train_gen.flow_from_directory(
    '../datasets/clothing-dataset-small-master/train', 
    target_size=(150, 150), 
    batch_size=32) # Found 3068 images belonging to 10 classes.

# see the types of classes present
# the classes correspond to the folder names in the clothing-dataset-small/train.
# this is inferred from the folder structure
train_ds.class_indices 

# OUTPUT
# {'dress': 0,
#  'hat': 1,
#  'longsleeve': 2,
#  'outwear': 3,
#  'pants': 4,
#  'shirt': 5,
#  'shoes': 6,
#  'shorts': 7,
#  'skirt': 8,
#  't-shirt': 9}
```

__High-level Overview of Building the Model__

<img src="08-images/Screenshot 2025-12-05 at 4.38.41 PM.png">

- after specifying the train and validation datasets, we create the base model (the convolutional layers). The base model used is Xception.
- in our case we should not include the dense layers. One does this by using `include_top=False` to exclude the dense layers. Keras thinks about CNNs as the convolutional layers being on the bottom and the dense layers being on the top ergo we state `include_top=False`.

<img src="08-images/Screenshot 2025-12-05 at 4.42.47 PM.png">

- To specify that we want to keep the convolutional layers, but we want to replace the dense layers we use the code below.

```python
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# we do not want to train a new model. We want to just have the convolution layers so we specify base_model.trainable = False
# this freezes the convolutional layers
base_model.trainable = False
```

- The next step is to create new dense layers. This is done by (1) defining the inputs that go into the base model. (2) Putting these inputs into the base model. (3) Grabbing the inputs from the base model and gathering the predictions.
    - define the inputs: `inputs = keras.Input(shape=(150, 150, 3))`
    - define the base model: `base = base_model(inputs, training=False)`
        - __VERY IMPORTANT to have training=False as an argument otherwise this can affect the validation accuracy__
    - Declare that base is the final model outputs = base
    - Defines what is fed into the model and what is outputted from the model `model = keras.Model(inputs, outputs)`
    - Then gather the model predictions: `preds = model.predict(X)` with the outputs of the model being stored in `preds`

```python
inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs, training=False)
outputs = base
model = keras.Model(inputs, outputs)

preds = model.predict(X)
preds.shape # (32, 5, 5, 2048)
```

- The above takes the 32 images and puts them into a base model. The base model is of dimension 32 x 5 x 5 x 2048 with an individual image having dimension 5 x 5 x 2048. This is because the CNN extracted the major features providing an object of 5 x 5 x 2048 for each image as the representation.
- Next, take this 5 x 5 x 2048 object and average each slice leaving one with a 32 vectors of length 2048.


<img src="08-images/Screenshot 2025-12-08 at 10.38.57 AM.png">

In code this pooling procedure looks like:

<img src="08-images">

#### Aside: Discussion of Downsampling, Pooling, and Depth Expansion

1. What is downsampling? - `Downsampling` is the processing of reducing the spatial dimensions  (height x width) of the image as it moves through the CNN. 
    - Downsampling is necessary because the number of calculations required without it would take too long.
    - Feature Hierarchy: We want the model to stop caring about pixels and start caring about the major features. By shrinking the images, the model looks at larger areas of the image at once.
2. Standard Pooling - `Pooling` is the means by which the dimension of the image is reduced. There are two major types of pooling:
    - Standard Pooling - involves creating a sliding window sliding over the 150 x 150 image. It looks at 4 pixels and keeps the strongest feature, but keeping the highest rated pixel value. This happens repeatedly until the image goes from 150 x 150 to 5 x 5.
    - Global Average Pooling - Take a 5 x 5 grid of pixels. The values in that grid are averaged to a single number reducing from 5 x 5 to a 1x1 pixel encoding the 5 x 5 information. This is repeated for the entire image. 
3. Depth Expansion: As the image gets smaller, we expand the number of channels/depth. One goes from 3 to 2048.
    - Starts out with a depth of 3 for the 3 channels in the image
    - As the network learns it creates new channels (filters).
    - Finally, we end up with a 5x5x2048 box in the drawing. With 2048 being the number of distinctive feature maps.
    - End result is a smaller image, but deeper information through the 2048 channel describing the different objects in the image.

All of the steps from downsampling to pooling to depth expansion:
1. Start with an image input of a tshirt.
2. The base is a 4D object with length x width x height x num_of_images
3. The base object undergoes pooling outputting a vector for each image.

#### End of Aside: 

The code below shows the logic and the transformations being applied.
1. First, take all of the images as inputs while defining the dimensions of the images.
2. These images are put into the base model where they are converted to a 32 x 5 x 5 x 2048 dimensional object
3. Pooling is then used on all 32 images with the average of each 5 x 5 slice taken. This results in 32 total vectors of 2048 length. This happens via `vectors = keras.layers.GlobalAveragePooling2D()(base)`
4. Since we are not using the original classes on imagenet, we train a new dense layer with 10 classes. The vectors from the previous operation are applied as the training data to generate the new dense layers.
5. Finally, we grab the model output.

- Looking at the predictions from the model, we get a nonsense model with the predictions not making much sense. When one uses `outputs = keras.layers.Dense(10)(vectors)` it just uses random numbers as the weights for the vectors. This obviously will not give classification results that make sense. 

- One must train the dense layer. To train the model we need a few key items:
    1. We need an optimizer that trains the weights used in the dense layer. The optimizer works by changing the elements of the weights matrix until we get a better solution over many iterations. The weights are changed so that the weights change so that the model learns. The optimizer used in this case is `optimizer = keras.optimizers.Adam(learning_rate=0.01)`. 
    2. Next, the optimizer needs to know how to change the weights for the model to improve. This uses a loss function. Depending on the type of learning problem we may use binary classification loss function, multiclass classification loss function or a regression loss function such as Mean Squared Error. In this case, use `loss = keras.losses.CategoricalCrossentropy(from_logits=True)`
        - It is a good idea to use `from_logits=True` as SOFTMAX takes input and turns it into a probability. Since we are dealing with probabilities we specify that `from_logits=True`
    3. Next, we compile the model. This means that it will show us model training and progress. Use: `model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    4. Finally, train the model.
        - Specify the number of epochs (the number of times the model trains on all batches that make up the training data.) Multiple batches make up the training data, and one round of training on all batches is one `epoch`. If we have 10 epochs this means we train the model 10 times on the data. `epochs=10`
        - Specify, the validation data to look at model accuracy and adjust any hyper parameters with `validation_data=val_ds`


```python
# create the base model
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(150, 150, 3))

base = base_model(inputs, training=False)

vectors = keras.layers.GlobalAveragePooling2D()(base)

outputs = keras.layers.Dense(10)(vectors)

model = keras.Model(inputs, outputs)

learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# records performance of training and validation 
history = model.fit(train_ds, epochs=10, validation_data=val_ds)
```


# 8.6 Adjusting the learning rate


The `learning rate` determines how well the model will fit the data. If the rate is too high, then we will probably see overfitting, if it is too low, the model will learn slowly resulting in underfitting. The goal is to find the rate the optimizes validation dataset performance.

Wrap the model code from above with the parameter `learning_rate`
```python
def make_model(learning_rate=0.01):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable=False

    ##########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(10)(vectors)
    model = keras.Model(inputs, outputs)

    ##########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model
```

Here is the code for training the model. Choose the `learning rate` that on average has the highest model performance while accounting for training time. One should use `validation dataset` for the learning rate. 
```python
# have the code for training
for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(lr)
    model = make_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[lr] = history.history

# plot the results
for lr, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=lr)

plt.xticks(np.arange(10))
plt.legend()
```


# 8.7 Checkpointing

Checkpointing is a way of saving a model after each iteration if certain conditions are met. For example, if a model achieves a certain validation performance we may setup a checkpoint that saves said model. `Callbacks` allow us to grab the different models and use them.

- `Callbacks` can be invoked with `keras.callbacks.ModelCheckpoint('<filename>.h5')`

<img src="08-images/Screenshot 2025-12-08 at 10.33.37 PM.png">


The code to save a checkpoint
```python
model.save_weights('model_v1.h5', save_format='h5') #save the model weights in a file

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# retrain the model but with a callback added
learning_rate=0.001

model = make_model(learning_rate=learning_rate)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[checkpoint]
) # saves models if they have better performance using the callbacks argument
```


# 8.8 Adding More Layers to the Neural Network

This is focused on adding more layers after one has created the vector representation. This means we add more layers to the dense layer.

Here is the current model we have with a vector representation with a dense layer.

<img src="08-images/Screenshot 2025-12-08 at 10.48.10 PM.png">


- Here we add another dense layer. The additional dense layers allows for the model to find feature combinations that can go together. Additionally, it helps with non-linear relationships.

- Another important feature is the activation functions that are at the end of each layer. The two activation layers used are ReLU and Logit function.
    - ReLU in first dense layer. ReLU states that if a value is negative make it zero, and if positive keep it.
    - The Logit function converts the values from the second layer into probabilities. (sigmoid or softmax)
    - There are many different activation functions

<img src="08-images/Screenshot 2025-12-08 at 10.55.14 PM.png">

The new model looks like the below:
```python
# Function to define model by adding new dense layer
def make_model(learning_rate=0.01, size_inner=100): # default layer size is 100
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(150,150,3))

    base_model.trainable = False
    
    #########################################
    
    inputs = keras.Input(shape=(150,150,3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors) # activation function 'relu'
    outputs = keras.layers.Dense(10)(inner)
    model = keras.Model(inputs, outputs)
    
    #########################################
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    
    return model
```

Next, train the model with different sizes of inner layers:
```python
# Experiement different number of inner layer with best learning rate
# Note: We should've added the checkpoint for training but for simplicity we are skipping it
learning_rate = 0.001

scores = {}

# List of inner layer sizes
sizes = [10, 100, 1000]

for size in sizes:
    print(size)
    
    model = make_model(learning_rate=learning_rate, size_inner=size)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[size] = history.history
    
    print()
    print()
```

One if using a GPU should monitor how training is happening on the GPU especially in online services such as Amazon Sagemaker
```bash
nvidia-smi #shows how much the gpu is being used

watch nvidia-smi # watch the gpu during training
```


# 8.9 Regularization and Dropout

During training, we sometimes have features that can appear on t-shirts, but are not central to defining a shirt. A chest pocket would be an example. This could cause the NN to assume that t-shirts must have this feature when they are really optional.

If a model is trained on 10 epochs, the network will see the chest pocket 10 times and assume this is a feature. 

We can use something called `dropout` where we randomly hide part of the image (NOT REALLY) allowing for the model to learn that a feature like a chest pocket are not essential to a t-shirt.

Here is a vector representation of the image with red areas highlighting the different parts of the vector that encode features.
<img src="08-images/Screenshot 2025-12-10 at 4.12.25 PM.png">


- What is actually happening is that in the dense layer a specific part of the inner layer is frozen. This forces the NN to focus on the other features of the network rather then other features of the network. This allows for the core shape to be discovered by the NN, and for superflous details like chest pockets to be ignored

<img src="08-images/Screenshot 2025-12-10 at 5.39.33 PM.png">

- When a part of the neural network (NN) is frozen the output still gets all of the layers. This means the output still gets to see the frozen feature. 

<img src="08-images/Screenshot 2025-12-10 at 5.43.59 PM.png">

- To incorporate this idea of dropout we develop a new model.
- The `dropout` portion will be included in the dense layer. Adding this dropout is also called `regularization`. It prevents the NN from overfitting to patterns which may be common, but are not necessary in describing a shirt.
- Here is the code with `regularization` added.
```python
def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    ###############################################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner) # apply the dropout to the final output of the inner layer

    outputs = keras.layers.Dense(10)(drop) #outputs gets the output of dropout

    model = keras.Model(inputs, outputs) # train the model

    ###############################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model
```

- Added regularization gives the below model workflow
<img src="08-images/Screenshot 2025-12-10 at 6.15.27 PM.png">

- Tuning dropout is the same as modifying any hyperparameter
```python
learning_rate = 0.01
size = 100

scores = {}

for droprate in [0.0, 0.2, 0.5, 0.8]:
    print(droprate)

    model = make_model(
        learning_rate=learning_rate,
        size=size,
        droprate=droprate
    )

    history = model.fit(train_ds, epochs=30, validation_data=val_ds)
    scores[droprate] = history.history

    print()
    print()


#########################################################################

#examine the dropout rate performance
for droprate, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % droprate)) # look at val performance
    plt.plot(hist['accuracy'], label=('train=%s' % droprate)) # look at train performance

plt.ylim(0.78, 0.86)
plt.legend()
plt.show()

```


# 8.10 Data Augmentation

- There is another way of addressing the overfitting that can occur with models is using `data augmentation` which means we add more training data by transforming the original data. For example, a shirt categorization service may have customers who send in pictures of shirts upside down, at angles etc. 

- `Data augmentation` allows us to take our current training data and transform it through rotations, and flipping thereby giving the model the necessary training data to still classify objects even if it is outside the norm.

- Keras has libraries for rotating, transforming, and changing images to allow for easy data augmentation.

- Adding augmentation is like any hyper-parameter. We can train a model with or without data augmentation for a few epochs and see if there is an increase in performance on the validation dataset. If we do not see an improvement, then throwout using data augmentation, if it worsens, then do not use it.

Here are possible image transformations:
- flip
- rotate
- shift
- shear
- zoom transformation
- one can also combine multiple transformations
<img src="08-images/Screenshot 2025-12-10 at 6.37.06 PM.png">

<img src="08-images/Screenshot 2025-12-10 at 6.42.01 PM.png">

- this can be done using:
```python
# transformation being used
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,  # selects a random number between -30 and 30 and rotates it
    width_shift=0.0,
    height_shift=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False
)


train_ds = train_gen.flow_from_directory(
    '../datasets/clothing-dataset-small-master/train',
    target_size=(150, 150),
    batch_size=32,
)

# DO NOT ADD augmentation to validation data
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    '../datasets/clothing-dataset-small-master/train',
    target_size=(150, 150),
    batch_size=32,
    shuffle=True
)
```
- DO NOT ADD AUGMENTATION  to validation data, only add AUGMENTATION to training data. One can think of the validation data as being images users are uploading in the wild. Data augmentation applied to the training data is trying to account for the various types of images users could upload. But applying data augmentation to validation data would first make comparing it against previous models that did not use data augmentation difficult. Additionally, validation data should not be influenced in such a way that it boosts performance.

How does one know when to use data augmentation and the type of data augmentation. It depends. Below are quick guidelines on when to use data augmentation
<img src="08-images/Screenshot 2025-12-11 at 6.16.47 PM.png">


Apply hyperparameter tuning to data augmentation:
__NOTE: As data augmentation means we add more data we need to increase the number of epochs for the model to be properly trained.__

```python
learning_rate=0.01
size=100
droprate=0.2

model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate)

# increase the number of epochs to 50 to account for the larger amount of training data
history = model.fit(train_ds, epochs=50, validation_data=val_ds)
```

### Aside: On maximizing the GPU and CPU for training and data augmentation

- During regular training we see that the GPU is fully utilized meaning the speed of training is faster. However, during data augmentation the CPU not the GPU is being used to create the images thereby slowing down the entire pipelines process. This process of the CPU processing the data augmentation and the GPU handling training is then repeated every iteration leading to a slowdown in the pipeline
<img src="08-images/Screenshot 2025-12-11 at 6.33.48 PM.png">

- One can speed up this pipeline by setting it up to have the CPU do the augmentation step while the GPU runs the training set when using multiple batches for processing.
- Check the `tensorflow.data` page to see how one can setup this process. Keras does not have the ability to perform this type of processing.
<img src="08-images/Screenshot 2025-12-11 at 7.09.16 PM.png">

Looking at the model validation accuracy 


# Training a larger model
- Train a larger 299x299 model 

__Some important notes about training a larger model__
- Since we are using data augmentation we are increasing the number of photos in the training data, we need to also expand the number of training epochs so that the model can properly learn from the data
- When training, if one sees large swings in validation performance this suggests that `learning_rate` is too high. In the below script the `learning_rate` is changed from `learning_rate=0.01` to `learning_rate=0.0005` which helps smooth out the jumps in validation performance.


```python
# function for creating the model
def make_model(input_size=150, learning_rate=0.01, size_inner=100, droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)

    base_model.trainable = False

    ################################################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(10)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model  
    )

# update the input size to train a larger model
input_size=299

# prepare the train and validation dataset

# augment the data for regularization
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_ds = train_gen.flow_from_directory(
    './clothing-dataset-small/train',
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    './clothing-dataset-small/validation',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)

# setup a checkpoint so that we can find a highly performant model during training
checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# train the model
learning_rate = 0.0005
size = 100
droprate = 0.2

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])
```

# 8.12 Using the Model
- loading the model
- Evaluating the model
- Getting predictions

```python
import tensorflow as tf
from tensorflow import keras

# load the model
keras.model.load_model('xception_v4_1_13_0.903.h5')

# grab some necessary imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

# test the model
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    '../datasets/clothing-dataset-small-master/test/',
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)

#evaluate the model
model.evaluate(test_ds)   # produces this as output [0.25407809, .908602]. The first number is the loss valude while the second number is accuracy of the model.

```


# 8.13 Summary

- Start by wanting to classify photos customers upload to our service to speed up the listing process. We use a neural network as it is especially suited for handling image data.
- To train the neural network we use tensorflow and keras. Keras is the higher level library built on-top of tensorflow. This is what we actually use when developing neural network models.

<img src="08-images/Screenshot 2025-12-13 at 11.47.37 PM.png">


- Most neural network are already trained and then made available for our use. Instead of building these models as is we can use a model that someone already trained. In keras there are two major layers:
    1. The bottom layer - is composed of the convolutional neural network which creates a vector representation of images we provide to the CNN
    2. The top layer - is the dense layer with the weights and classes. This is the layer we modify and change providing our own classes and generating our own weights so that we can classify images properly with classes we want.

- After this one should focus on tuning major hyperparameters:
    1. learning rate - how fast the model trains
    2. dense layer size - the size of the dense layer
    3. droprate - is a form of regulatization which in a sense "blocks off parts of the vector representation" provided to the dense layer which prevents overfitting of superfluous features which are not essential for classification.
    4. Data augmentation - is a way of preventing overfitting my transforming training images to better match the types of photos users may submit. 

- Checkpointing is also included as a way to save a model that performs well during training.



