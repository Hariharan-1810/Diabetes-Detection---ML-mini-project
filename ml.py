import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('diabetes.csv')

#df.head()

# Step 3: Check for missing values
df.isnull().sum()

# Step 4: Split the data into features and labels
x = df.iloc[:, df.columns != 'Outcome']
y = df.iloc[:, df.columns == 'Outcome']

# Step 5: Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Step 6: Train the model
model = RandomForestClassifier()
model.fit(xtrain, ytrain.values.ravel())

# Step 7: Make predictions
predict_output = model.predict(xtest)

# Step 8: Evaluate the model with additional metrics
precision = precision_score(ytest, predict_output)
recall = recall_score(ytest, predict_output)
f1 = f1_score(ytest, predict_output)
conf_matrix = confusion_matrix(ytest, predict_output)

# Print additional metrics
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', conf_matrix)

# Step 9: Plot a graph
plt.figure(figsize=(10, 6))

# Example: Plotting Glucose vs. BMI with Outcome as hue
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, palette='coolwarm', s=100)

# Set the plot labels and title
plt.title('Glucose vs. BMI - Diabetes Outcome')
plt.xlabel('Glucose Level')
plt.ylabel('BMI')

# Save the plot as an image
plt.savefig('diabetes_outcome_plot.png')

# Display the plot
plt.show()

# Step 10: Print accuracy
acc = accuracy_score(predict_output, ytest)
print('The accuracy score for Random Forest:', acc)
