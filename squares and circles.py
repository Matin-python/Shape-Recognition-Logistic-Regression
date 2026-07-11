import numpy as np 

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

DATASET_SIZE = 20

image = []
for i in range(DATASET_SIZE):
    image.append(preprocess_image(f"dataset/{i}.jpg"))

image = np.array(image)


# 0 = Square    1 = Circle   
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

x_train, x_test, y_train, y_test = train_test_split(image, 
                                                    labels, 
                                                    test_size=0.2, 
                                                    random_state= 42)

reg_logistic = linear_model.LogisticRegression(max_iter=10000)
reg_logistic.fit(x_train, y_train)

out_pred = reg_logistic.predict(x_test)

# print(out_prod)
# print(y_test)


accuracy = sm.accuracy_score(y_test, out_pred)
print(f"Accuracy: {accuracy:.2%}")

msr = sm.mean_squared_error(y_test, out_pred)
print('mean squared error= ', msr)

test = preprocess_image("dataset/t2.jpg")

plt.imshow(Image.open("dataset/t2.jpg"))
plt.axis("off")
plt.show()

print()
out_test = reg_logistic.predict(test.reshape(1, -1))
if out_test[0] == 0:
    print("Prediction: Square")
else:
    print("Prediction: Circle")
