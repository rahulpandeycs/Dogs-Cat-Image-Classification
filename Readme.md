
# Details

The project contains modified files from fastAi course(Practical Deep Learning for Coders) lesson 1 for dogs and cat problem. It uses famous Pretrained VGG model with our Vgg16 class and reduces already exisiting 1000+ classes to predict 2 classes relevant for us, i.e Cats and Dogs.

Since 'cat' and 'dog' are not categories in Imagenet - instead each individual breed is a separate category.

To change our model so that it outputs "cat" vs "dog", instead of one of 1,000 very specific categories, we need to use a process called "finetuning". Finetuning looks from the outside to be identical to normal machine learning training - we provide a training set with data and labels to learn from, and a validation set to test against. The model learns a set of parameters based on the data provided. However, the difference is that we start with a model that is already trained to solve a similar problem. The idea is that many of the parameters should be very similar, or the same, between the existing model, and the model we wish to create. Therefore, we only select a subset of parameters to train, and leave the rest untouched. This happens automatically when we call fit() after calling finetune().


For more explanation please refer: http://wiki.fast.ai/index.php/Lesson_1_Notes 

## Project setup:

Prerequistes for course: python 3+, keras 2+, tensorflow

Download datasets from: http://files.fast.ai/data/ , in our case we download dogscats.zip. Extract this in the project data folder



Why this repo ?

The project created by fast ai team uses keras 1.x and python 2.x at the time of creation. But now if you use it over keras 2.x and python 3.x the project gives you error. The modifications to accomodate these changes has been done in this repo and the code contains all the changes as mentioned below after version change.

Reference: @atlas7 , http://forums.fast.ai/t/keras-2-released/1956/42

Notes on Python 2.x / Keras 1.x to Python 3.x / Keras 2.x transition
Some notes on moving from Python 2.x & Keras 1.x -> to Python 3.x & Keras 2.x. (Note Keras currently supports Python 2.7 to 3.5 only. i.e. Python 3.6 will not work on Keras - yet).

Change accordingly in vgg16.py and utils.py.

### Keras 1.x -> Keras 2.x
#### Keras 1.x:

1. from keras.layers.convolutional import Convolution2D 
2. from keras.regularizers import l2, activity_l2, l1, activity_l1
3. from keras.utils.layer_utils import layer_from_config
4. from keras import backend
5. Convolution2D
6. batches.nb_sample
7. batches.nb_class
8. model.add(Convolution2D(filters, 3, 3, activation="relu"))
9. fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch, validation_data=val_batches, nb_val_samples=val_batches.nb_sample)
10. nb_epoch
11. self.model.predict_generator(test_batches, test_batches.nb_sample)

#### Keras 2.x:

1. from keras.layers.convolutional import Conv2D
2. from keras.regularizers import l2, l1
3. from keras.layers import deserialize as layer_from_config
4. from keras import backend; backend.set_image_dim_ordering('th')
5. Conv2D
6. batches.samples
7. batches.num_class
8. model.add(Conv2D(filters, (3, 3), activation="relu"))
9. fit_generator(batches, steps_per_epoch=batches.samples//batches.batch_size, epochs=nb_epoch, validation_steps=val_batches.samples//val_batches.batch_size)
10. epochs
11. self.model.predict_generator(test_batches, test_batches.samples//test_batches.batch_size)

### Note:

8. goes from: “3, 3” … to “(3, 3)”… i.e. with the brackets.
9. in Keras 1, the progress bar shows total number of training samples processed. In Keras 2, the progress bar shows total number of batches processed (steps_per_epoch). Recall that total batches = total samples // batch size. (I floor it to ensure integer value).
11. in Keras 1, regarding model.predict_generator(), 2nd argument corresponds to number of total samples. In Keras 2, that 2nd argument becomes total number of batches. Recall that total batches = total samples // batch size. (I floor it to ensure integer value).
Note: it looks like Keras 1 talks in “number of samples”. Keras 2 prefers “number of batches”. Beware.

### Python 2.x -> Python 3.x
####Python 2.x:

1. import cPickle as pickle
2. reload()

#### Python 3.x:

1. import _pickle as pickle 
2. from importlib import reload; reload()
