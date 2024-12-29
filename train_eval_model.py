import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Dataset.csv')

oversample = SMOTE(random_state = 42)
x_resampled, y_resampled = oversample.fit_resample(df[['EAR', 'MAR']], df['Label'])

x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.3, random_state = 42)

model = RandomForestClassifier(n_estimators = 100, random_state = 42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels = ['Alert', 'Drowsy'], yticklabels = ['Alert', 'Drowsy'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model, 'Drowsiness-Model.pkl')

print("Training of Model is completed successfully.")