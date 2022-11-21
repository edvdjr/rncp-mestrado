import pandas
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import numpy as np

out_path = "out/"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available: ', len(physical_devices))
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], true)

# load training dataset (no previous training)

dataframe = pandas.read_csv(out_path + "training_params.csv", header=None)
dataset = dataframe.values
train_samples = dataset[:,0:-1].astype(float)
train_labels = dataset[:,-1]

train_labels = np.array([train_labels])
train_samples = np.array([train_samples])
train_labels, train_samples = shuffle(train_labels, train_samples)
train_labels = train_labels.T

n_features = 10
class_labels = [x for x in range(2)]
display_labels = ["Face", "Moto"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, n_features))

model = Sequential([
    Dense(units=32, input_shape=(n_features,), activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=2, activation='softmax')
])

print(scaled_train_samples.shape)
print(train_labels.shape)

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=100, shuffle=True, verbose=1)

model.save('out/model_caltech')

# End of training

# from tensorflow.keras.models import load_model
# new_model = load_model('models/mnist_model')

# load test dataset
dataframe = pandas.read_csv(out_path + "testing_params.csv", header=None)
dataset = dataframe.values
test_samples = dataset[:,0:-1].astype(float)
test_labels = dataset[:,-1]

test_labels = np.array([test_labels])
test_samples = np.array([test_samples])
test_labels, test_samples = shuffle(test_labels, test_samples)
test_labels = test_labels.T

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, n_features))

#Predictions
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

rounded_predictions = np.argmax(predictions, axis=-1)

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions, labels=class_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(values_format = '')
plt.savefig(out_path + 'CM.png', bbox_inches='tight')
# plt.show()

f = open(out_path + "classification_report", "w")
f.write(
    f"Classification report:\n"
    f"{metrics.classification_report(test_labels, rounded_predictions)}\n"
)
f.close()