import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

marks = {
    'Class': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Marks': [100, 95, 94, 99, 100, 96, 91, 90, 89]
}

df = pd.DataFrame(marks)

X = df[['Class']]
Y = df['Marks']

model = LinearRegression()
model.fit(X,Y)

marks_10 = np.array([[10]])
predicted_marks = model.predict(marks_10)

print(predicted_marks)

# Plot the actual data points
plt.scatter(df['Class'], df['Marks'], color='blue', label='Actual Marks')

# Plot the Regression Line
plt.plot(df['Class'], model.predict(X), color='red', linewidth=2, label='Regression Line')

# Mark the predicted point for the 10th student
plt.scatter(10, predicted_marks, color='green', marker='o', s=100, label=f'Predicted Marks: {predicted_marks[0]:.2f}')

# Labels and title
plt.xlabel("Class")
plt.ylabel("Marks")
plt.title("Student Marks Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()