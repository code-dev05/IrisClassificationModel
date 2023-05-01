import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Importing the Data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

# Encoding the Labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Formatting the data for manipulation
x = df.drop(columns=['Species'])
y = df['Species']
x = x.to_numpy()
y = y.to_numpy()

# Splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Training the Model with the training data
model = LogisticRegression(solver='liblinear', C=3, multi_class='ovr', random_state=0)
model.fit(x_train, y_train)

# Prediciting the labels using testing data as the input
y_pred = model.predict(x_test)

# Forming a confusion matrix to compare the true labels and the predicted labels
cm = confusion_matrix(y_test, y_pred)

# Creating two different windows for plotting
fig1 = plt.figure(figsize=(10, 10))
fig2 = plt.figure(figsize=(10, 10))

# Plotting the Confusion Matrix
fig1.canvas.manager.set_window_title('Confusion Matrix')
ax1 = fig1.add_subplot()
ax1.imshow(cm)
ax1.grid(False)
ax1.set_xlabel('Predicted outputs', fontsize=20, color='black')
ax1.set_ylabel('Actual outputs', fontsize=20, color='black')
ax1.xaxis.set(ticks=range(3))
ax1.yaxis.set(ticks=range(3))
ax1.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax1.text(j, i, cm[i, j], ha='center', va='center', color='white')

# Plotting the different histograms of the features
fig2.canvas.manager.set_window_title('Histograms of the Features')
ax2 = fig2.add_subplot(2, 2, 1)
ax2.hist(x[:, 0], bins=30)
ax2.set_xlabel('SepalLengthCm')
ax2.set_ylabel('Frequency')

ax3 = fig2.add_subplot(2, 2, 2)
ax3.hist(x[:, 1], bins=30)
ax3.set_xlabel('SepalWidthCm')
ax3.set_ylabel('Frequency')

ax4 = fig2.add_subplot(2, 2, 3)
ax4.hist(x[:, 2], bins=30)
ax4.set_xlabel('PetalLengthCm')
ax4.set_ylabel('Frequency')

ax5 = fig2.add_subplot(2, 2, 4)
ax5.hist(x[:, 3], bins=30)
ax5.set_xlabel('PetalWidthCm')
ax5.set_ylabel('Frequency')

plt.show()

print(f'Accuracy of Training Data: {model.score(x_train, y_train)*100: .2f}%')
print(f'Accuracy of Testing Data: {model.score(x_test, y_test)*100: .2f}%')
