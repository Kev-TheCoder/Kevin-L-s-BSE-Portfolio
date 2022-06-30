# Raspberry Pi Image Recognition
This application uses AI, or machine learning, and both tensorflow and tensorflow lite along with neural networks to categorize and/or classify any given image. 

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Kevin L | Dublin High School | Computer Science | Incoming Junior

# Software/Tools Requirement
- Raspberry Pi 4 Model B
- VNC Viewer
- PUTTY
- Micro SD card

Note: The following below is used in the code presented below
- Jupyter Notebook V6.4.11
- Tensorflow V2.9.1
- Numpy V1.23.0
- Opencv-python V4.6.0.66
- Scipy V1.7.3
- Matplotlib

# Code

```py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import scipy
import keras
train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)
train_dataset = train.flow_from_directory("C:\\Users\\kevin\\Fruit or Vegetable\\Testing", 
                                          target_size = (200, 200), 
                                          batch_size = 4, class_mode = 'binary')
validation_dataset = validation.flow_from_directory("C:\\Users\\kevin\\Fruit or Vegetable\\Testing",
                                            target_size = (200, 200), 
                                        batch_size = 4, class_mode = 'binary')
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape = (200, 200, 3)), 
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), 
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                                   ])
model.compile(loss= 'binary_crossentropy' ,
             optimizer = RMSprop(learning_rate = 0.001),
             metrics = ['accuracy'])                                   
##Training

model_fit = model.fit(train_dataset,steps_per_epoch = 1,
                     epochs = 100,
                     validation_data = validation_dataset)
dir_path = "C:\\Users\\kevin\\Fruit or Vegetable\\Validating"
for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '\\' + i, target_size = (200,200))
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis = 0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        print("Fruit")
    else:
        print("Vegetable")
##Renaming files in the folder

import os
import pathlib

i = 1000

path = os.chdir("C:\\Users\\kevin\\Fruit or Vegetable\\Validating")
for file in os.listdir(path):
    img = image.load_img(dir_path + '\\' + file, target_size = (200,200))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis = 0)
    images = np.vstack([X])
    val = model.predict(images)
   
    if val == 0:
        new_file_name = "Fruits{}.jpg".format(i)
        os.rename(file, new_file_name)
    else:
        new_file_name = "Vegetables{}.jpg".format(i)
        os.rename(file, new_file_name)
    
    i = i + 1
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis = 0)
    images = np.vstack([X])
    val = model.predict(images)
mobile = tf.keras.applications.mobilenet.MobileNet()
import os
dir_path = "C:\\Users\\kevin\\Fruit or Vegetable\\Validating"
for file in os.listdir(dir_path):
    img = image.load_img(dir_path + "\\" + file, target_size = (224, 224))

    resized_img = image.img_to_array(img)
    final_image = np.expand_dims(resized_img, axis = 0)
    final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)
    
    predictions = mobile.predict(final_image)
    results = imagenet_utils.decode_predictions(predictions)
    print(results)
    plt.imshow(img)
    plt.show()    
```

  
# Final Milestone
My final milestone is modifing my environment to be more applicable for the viewing audience, renaming files according to the classification using for loop, and providing the outcomes(probability) of each classification. My original goal of this milestone was to transfer my model to tensorflow lite which was unsuccessful as I encountered many errors. To save time and resources, I continued to use my environment, jupyter notebook, as my place to present my model and programmed the additional features: renaming files and the outcomes. Renaming files was a moderate task because the logic behind it was relatively simple but caused multiple errors, which one of them being that the file already existed. I was able to correct this error by writing a psudo-code, thinking on what needs to happen and the order. This became a success as files were being written correctly without errors. The outcomes was a moderate task as well because I had coded this eariler in a testing model while developing this current model. However, the outputs of this weren't the ideal predicitons I wanted which resulted in debugging and retraining.

<iframe width="800" height="450" src="https://www.youtube.com/embed/-L5Gj5mpwqY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Second Milestone

My second milestone was creating my first custom model. I downloaded different softwares to create the environment for my custom model to be coded, trained, and executed. The model that I created was classifying whether the image was a fruit or a vegetable. During this milestone, I also created a fruit classification model which could tell what type of fruit the given image was. Even though these were two seperate models, they both had the same concepts which interwined that helped me develop both of these models. I had a lot of trouble during this milestone, for example: setting up the environment, importing libraries and files, as well as some syntax errors. Also, I used tensorflow to develop both of these models as I had trouble running tensorflow lite. I plan to transfer these models into tensorflow lite which then I can use it in the VNC viewer.  

<iframe width="800" height="450" src="https://www.youtube.com/embed/5f2XdcS93u0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# First Milestone
  

My first milestone was setting up the VNC viewer, raspberry pi, and running my first model on VNC viewer. In order to run my first model, I had to set up tensorflow lite, a smaller version of tensorflow. Even though those two softwares were relatively similar, I had to adjust a lot of the applications towards tensorflow lite since raspberry pi no longer supports tensorflow. This gave me a lot of trouble setting up the first model but I succeeded. I took a pretrained model that was available from tensorflow.org and downloaded/imported the files into my VNC viewer. I also had to adjust code in the given pretrained model since it was adapted to tensorflow instead of tensorflow lite. As this was a pretrained model, the outputs were already predetermined but it marked the success and functionality of my application. I had many struggles during this time as all of these were new concepts to me but I had developed a working prototype. 

<iframe width="800" height="450" src="https://www.youtube.com/embed/xq-7jg8sWTU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

