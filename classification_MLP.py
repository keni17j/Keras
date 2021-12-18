"""Classification.
Keyword: Sequential,　MLP, MNIST.
In order to use plot_model, execute following codes.
(1) pip3 install pydot
(2) pip3 install graphviz
(3) brew install graphviz
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix


def main():
    """Main function."""

    # Load datas.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Show informations of datas.
    show_datas(x_train, y_train, x_test, y_test)

    # Preprocess datas.
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    # Create models.
    input_shape = x_train.shape[1]
    output_shape = y_train.shape[1]
    model = create_model(input_shape, output_shape)

    # Leaen models.
    dir_path = os.path.basename(__file__)[:-3]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    batch_size = 500
    epochs = 50
    learn_model(dir_path, model, x_train, y_train, batch_size, epochs)

    # Load the model.
    file_path = os.path.join(dir_path, 'model.h5')
    model = load_model(file_path)
    file_path = os.path.join(dir_path, 'model.png')
    plot_model(model, file_path, show_shapes=True)

    # Predict test datas.
    predict(model, x_test, y_test, dir_path)


def show_datas(x_train, y_train, x_test, y_test):
    """Show shapes, values, and images."""

    # Show shapes.
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    # Show a data value.
    #print(x_train[0])
    #print(y_train[0])

    #  Show an image.
    img = x_train[0]
    img = Image.fromarray(img)
    #img.show()


def preprocess(x, y):
    """When x is image data, convert to 0-1 from 0-255.
    And convert each data from two-dimensional array to one-dimensional array.
    y is converted to the one-hot vector.
    """

    # About x.
    x = x / 255
    x = np.reshape(x, (len(x), -1))

    # About y.
    y_unique = np.unique(y)
    y = to_categorical(y, y_unique.size)

    return x, y


def create_model(input_shape, output_shape):
    """Sequential model.
    MLP consists of Dense and Dropout.
    Optimizers: SGD, RMSprop, Adagrad, Adadelta, Adam, etc.
    """

    model = Sequential()
    model.add(Dense(512, input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(output_shape, activation='softmax'))

    loss = 'categorical_crossentropy'
    opt = optimizers.Adam(lr=0.01)  # Set the learning rate to optimizers.
    met = ['acc']
    model.compile(loss=loss,  # Used in learning.
                  optimizer=opt,
                  metrics=met,  # Not used in learning.
                  )

    model.summary()

    return model


def learn_model(dir_path, model, x_train, y_train, batch_size, epochs):
    """Learn the model.
    The best model is saved.
    The current model is not best.
    So, you have to load the best model later.
    """

    file_path = os.path.join(dir_path, 'model.h5')
    mcp = ModelCheckpoint(filepath=file_path,
                          verbose=1,
                          save_best_only=True,
                          )
    hist = model.fit(x=x_train,
                     y=y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     callbacks=[mcp],
                     validation_split=0.2,
                     shuffle=True,
                     )

    # Save histories as graphs.
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    file_path = os.path.join(dir_path, 'acc.png')
    graph(train_acc, val_acc, 'acc', file_path)
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    file_path = os.path.join(dir_path, 'loss.png')
    graph(train_loss, val_loss, 'loss', file_path)


def graph(train, val, y_label, file_path):
    """make graphs."""

    # Create a figure.
    fig = plt.figure()
    # Titles.
    plt.xlabel('epochs', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    # Scale dierctions.
    plt.tick_params(which='both', direction='in')
    # Plot datas.
    plt.plot(train,
             label='train',
             )
    plt.plot(val,
             label='valid',
             )
    # Limit the range.
    plt.xlim(0,)
    plt.ylim(0,)
    # Legends.
    plt.legend(loc='upper right', ncol=1, fontsize=10)
    # Save.
    fig.savefig(file_path)


def predict(model, x_test, y_test, dir_path):
    """Predict test datas.
    Show results.
    """

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cmx = confusion_matrix(y_true, y_pred)
    print(cmx)
    acc = np.count_nonzero(y_pred==y_true) / len(y_pred) * 100
    acc = '{:.2f}'.format(acc)
    print('Accuracy', acc, '%')

    # Save results as a csv.
    num = np.unique(y_true).size
    index = np.arange(num)
    columns = np.arange(num)
    df = pd.DataFrame(cmx, index=index, columns=columns)
    file_path = os.path.join(dir_path, 'predict.csv')
    df.to_csv(file_path)
    df = pd.DataFrame([acc], index=['Accracy'])
    df.to_csv(file_path, mode='a', header=False)


if __name__ == '__main__':
    main()
