I try to the deep learning using Keras.

自作ではラベルは数字になっていないのでこれを使う
    for i, val in enumerate(y_unique):
        print(i, ':', val)
        y = np.where(y==val, i, y)




In order to use plot_model, execute following codes.
(1) pip3 install pydot
(2) pip3 install graphviz
(3) brew install graphviz