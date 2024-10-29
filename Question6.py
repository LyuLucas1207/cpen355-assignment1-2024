from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, data_home='./Data')
# change the data type to numpy array
X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)
# First Step: Split the dataset into training set and temporary set (70% training set, 30% temporary set)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=520)
# Second Step: Split the temporary set into validation set and test set (20% validation set, 10% test set)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=520)
# initialize the model
svc = SVC(kernel='rbf', C=1.0, gamma='scale')
# train the model
print('Training the model...')
svc.fit(X_train, y_train)
# Output the training accuracy
train_accuracy = svc.score(X_train, y_train)
print(f'Training Accuracy: {train_accuracy:.2f}')

"""
kernel: kernel function, default='rbf'
        linear: this is just the dot product between any two given observations. No transformation is done.
        poly: this is the polynomial kernel. It is used when the data is not linearly separable.
        rbf: this is the radial basis function kernel. It is used when the data is not linearly separable.
        sigmoid: this is the hyperbolic tangent kernel. It is used when the data is not linearly separable.
C: regularization parameter, default=1.0
gamma: kernel coefficient, default='scale'
        scale: 1 / (n_features * X.var())
        auto: 1 / n_features
"""