Using TensorFlow backend.
26677
21341
5336
model2.py:98: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(24, (5, 5), activation="elu", strides=(2, 2))`
  model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
model2.py:100: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(36, (5, 5), activation="relu", strides=(2, 2))`
  model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model2.py:103: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(48, (5, 5), activation="elu", strides=(2, 2))`
  model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
model2.py:106: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(64, (3, 3), activation="relu")`
  model.add(Convolution2D(64,3,3, activation='relu'))
model2.py:109: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(96, (3, 3), activation="elu")`
  model.add(Convolution2D(96,3,3, activation='elu'))
model2.py:133: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  nb_val_samples=len(validation_samples), nb_epoch=10, callbacks=callbacks_list, verbose=1)
model2.py:133: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<generator..., verbose=1, steps_per_epoch=21341, epochs=10, callbacks=[<keras.ca..., validation_data=<generator..., validation_steps=5336)`
  nb_val_samples=len(validation_samples), nb_epoch=10, callbacks=callbacks_list, verbose=1)
Epoch 1/10
2018-08-27 08:21:43.422411: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-08-27 08:21:43.422465: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-08-27 08:21:43.422491: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-08-27 08:21:43.515077: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-08-27 08:21:43.515465: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties:
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.09GiB
2018-08-27 08:21:43.515501: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0
2018-08-27 08:21:43.515518: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y
2018-08-27 08:21:43.515534: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
21340/21341 [============================>.] - ETA: 0s - loss: 0.0479 - acc: 0.5825Epoch 00000: val_acc improved from -inf to 0.58893, saving model to weights-improvement-00-0.59.hdf5
21341/21341 [==============================] - 440s - loss: 0.0479 - acc: 0.5825 - val_loss: 0.0449 - val_acc: 0.5889
Epoch 2/10
21339/21341 [============================>.] - ETA: 0s - loss: 0.1486 - acc: 0.5895Epoch 00001: val_acc improved from 0.58893 to 0.59152, saving model to weights-improvement-01-0.59.hdf5
21341/21341 [==============================] - 445s - loss: 0.1486 - acc: 0.5896 - val_loss: 0.0446 - val_acc: 0.5915
Epoch 3/10
21340/21341 [============================>.] - ETA: 0s - loss: 0.0649 - acc: 0.5854Epoch 00002: val_acc did not improve
21341/21341 [==============================] - 445s - loss: 0.0649 - acc: 0.5854 - val_loss: 0.0768 - val_acc: 0.5852