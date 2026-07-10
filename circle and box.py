import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn import linear_model 
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_image(path):
    """
    Load an image, convert it to grayscale,
    flatten it into a 1D feature vector,
    and normalize pixel values.
    """
    image = Image.open(path)
    image = np.array(image)
    image = image[:, :, 0]
    image = image.flatten()
    image = image / 255
    return image

image = []
for i in range(20):
    image.append(preprocess_image(f"dataset/{i}.jpg"))

image = np.array(image)


# 0 = Square    1 = Circle 
real_out = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

x_train, x_test, y_train, y_test = train_test_split(image, 
                                                    labels, 
                                                    test_size=0.2, 
                                                    random_state= 42)

reg_logestic = linear_model.LogisticRegression(max_iter=10000)
reg_logestic.fit(x_train, y_train)

out_prod = reg_logestic.predict(x_test)

# print(out_prod)
# print(y_test)

#cor = 0
#incor = 0
#for i in range(y_test.size):
#    if out_prod[i] == y_test[i]:
#        cor += 1
#    elif out_prod[i] != y_test[i]:
#        incor += 1

# print (cor, incor)
# print(y_test.size)

#correct_percentage = (cor * 100) / y_test.size
#print("correct prediction= ", correct_percentage, "%")

accuracy = sm.accuracy_score(y_test, out_pred)
print(f"Accuracy: {accuracy:.2%}")

msr = sm.mean_squared_error(y_test, out_prod)
print('mean squared error= ', msr)


test = Image.open(f'dataset/t2.jpg')
test= np.array(test)
test = test[:, :, 0]
test = test.flatten()
test = test / 255

plt.imshow(Image.open("dataset/t2.jpg"))
plt.axis("off")
plt.show()

out_test = reg_logestic.predict(test.reshape(1, -1))
if out_test[0] == 0:
    print("Prediction: Square")
else:
    print("Prediction: Circle")
