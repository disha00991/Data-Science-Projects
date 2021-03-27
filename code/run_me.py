import numpy as np
from matplotlib import pyplot as plt

from TreeClassifier import question_4

stuff=np.load("../report_src/data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_tst = stuff["X_tst"]

def show(x):
    img = x.reshape((3,32,32)).transpose(1,2,0)
    plt.imshow(img)
    plt.axis('off')
    plt.draw()
    plt.pause(10)

####Q:4 Train 6 different classification trees on the image data, with each of the following maximum depths
####{1,3,6,9,12,14}
####use 5-fold cross validation
question_4(X_trn, y_trn, X_tst)




