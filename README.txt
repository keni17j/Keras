I try to the deep learning using Keras.

自作ではラベルは数字になっていないのでこれを使う
    for i, val in enumerate(y_unique):
        print(i, ':', val)
        y = np.where(y==val, i, y)
