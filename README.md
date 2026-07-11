# Shape Recognition using Logistic Regression

A simple Machine Learning project that classifies images as **Squares** or **Circles** using **Logistic Regression**. The model is trained on a manually created dataset and demonstrates the complete image classification workflow, including image preprocessing, model training, evaluation, and prediction.

## Features

* Binary image classification (Square vs. Circle)
* Manual image preprocessing
* Image normalization
* Train/test dataset splitting
* Logistic Regression classifier
* Model evaluation using accuracy and mean squared error
* Predict the class of a new image
* Easy to extend with additional shapes or larger datasets

## Technologies Used

* Python 3
* NumPy
* Scikit-learn
* Pillow (PIL)
* Matplotlib

## Project Structure

```text
Shape-Recognition-Logistic-Regression/
│
├── dataset/
│   ├── 0.jpg
│   ├── ...
│   ├── 19.jpg
│   └── t2.jpg
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
│
└── images/
    ├── dataset_examples.png
    └── prediction_example.png
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Matin-python/Shape-Recognition-Logistic-Regression.git
cd Shape-Recognition-Logistic-Regression
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

## Dataset

The project uses a manually created dataset consisting of:

* 10 Square images (Label = 0)
* 10 Circle images (Label = 1)

Each image is:

* Loaded using Pillow
* Converted to grayscale
* Flattened into a one-dimensional feature vector
* Normalized to pixel values between 0 and 1

## Usage

Run the project:

```bash
python main.py
```

The program will:

1. Load and preprocess the dataset.
2. Split the data into training and testing sets.
3. Train a Logistic Regression classifier.
4. Evaluate the model.
5. Predict the shape of a new image (`dataset/t2.jpg`).

## Example Output

```text
Accuracy: 100.00%
Mean Squared Error: 0.0

Prediction: Circle
```

> **Note:** The reported accuracy may vary because of the small dataset size.

## How It Works

1. Load the training images.
2. Convert each image into a grayscale feature vector.
3. Normalize the pixel values.
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression model.
6. Evaluate the model using accuracy and mean squared error.
7. Predict the class of a new image.

## Example Images

### Training Dataset

Add a screenshot of several training images here.

### Test Image

Add a screenshot of the image used for prediction.

## Future Improvements

* Increase the dataset size
* Add more geometric shapes
* Support color images
* Perform automatic image resizing
* Display prediction confidence
* Add a confusion matrix
* Compare Logistic Regression with other machine learning algorithms
* Implement the same project using Deep Learning (CNN)

## Related Projects

A Deep Learning version of this project will also be available, demonstrating the same classification task using a neural network for comparison.

## License

This project is licensed under the MIT License.

## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Computer Vision, Machine Learning, Deep Learning, and Artificial Intelligence.
