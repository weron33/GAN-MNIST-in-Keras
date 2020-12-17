# GAN-MNIST-in-Keras
Here I created a GAN model. It is able to generate images simmilar to those from MNIST dataset. 
During developing this project I found lack of class-build codes that solve this task, so here I created one to give You an example how it can be done. 

So, it contains 4 classes:
  - BachtedData class, which is resposible for downloading, batching and choosing data.
  - Generator class, that contains generator model.
  - Discriminator class, where discriminator is defined.
  - AdversarialModel class, which connects discriminator and generator in one model and also set as an input to it data form BatchedData class.

BatchedData contains:
  - next_batch method, to generate next set of data.

Most of function are written in AdversarialModel, it contains:
  - compile method, to compile model with Adam optimizer, binary crossentropy loss function and accuracy metrics. It also creates new discriminator.
  - predict method, to predict if passed images are real or fake.
  - generate_images method, to generate fake images from 
  - train method, to train model on a batches.
  - save method, to save model in filepath.
  - load method, to load model from filepath.
  - plot method, to plot models' train history.

File with name main.py is a some sort of workspace. So it shows an example how to call my classes. All changes in models should be prepared in ml_model.py.

Feel free to modify and use this code as much as You like.
