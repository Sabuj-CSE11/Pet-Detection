# Pet-Detection
Cats and Dogs detection for the LfD course using CNN.

Dataset: https://www.kaggle.com/datasets/imenbakir/cat-and-dog


CNN Model Architecture:

conv_model = Sequential()

conv_model.add(Input(shape=(180, 180, 3)))
conv_model.add(Conv2D(16, kernel_size=(3, 3), activation="relu"))

conv_model.add(MaxPooling2D(pool_size=(2, 2)))
conv_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))

conv_model.add(MaxPooling2D(pool_size=(2, 2)))
conv_model.add(Dropout(0.5))

conv_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
conv_model.add(MaxPooling2D(pool_size=(2, 2)))

conv_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
conv_model.add(MaxPooling2D(pool_size=(2, 2)))

conv_model.add(Flatten())
conv_model.add(Dropout(0.5))
conv_model.add(Dense(64, activation="sigmoid"))
conv_model.add(Dense(1, activation="sigmoid"))
     
