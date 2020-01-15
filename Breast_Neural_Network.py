from keras import layers
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Data
data = pd.read_csv('PATH')
x = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Model
model = Sequential()
# Input layer
model.add(layers.Dense(150, activation='sigmoid', input_dim=5))
# Hidden layer
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='sigmoid'))
# Output layer
model.add(layers.Dense(1, activation='sigmoid'))
# Compilation
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)
# Training
model.fit(x_train, y_train, epochs=100)
# Accuracy
print(model.evaluate(x_test, y_test))
