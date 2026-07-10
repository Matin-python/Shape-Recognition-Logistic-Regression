import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn import linear_model 
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image


image = []
for i in range (20):
    image.append(Image.open(f'dataset/{i}.jpg'))
    image[i] = np.array(image[i])
    image[i] = image[i][:, :, 0]
    image[i] = image[i].flatten()
    image[i] = image[i] / 255


# 0 = Square    1 = Circle 
real_out = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

x_train, x_test, y_train, y_test = train_test_split(image, 
                                                    labels, 
                                                    test_size=0.2, 
                                                    random_state= 42)

reg_logestic = linear_model.LogisticRegression()
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

out_test = reg_logestic.predict(test.reshape(1, -1))
print(out_test)
