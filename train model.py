import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load the dataset
df = pd.read_csv(r"D:\Nareshit Videos and notes\Omkar Sir-FSDS & GenAI\FSDS&GENAI_Notes\Projects\FastAPI\smp_data_from_app.csv")

# 2. Select correct columns (as per your CSV)
X = df[["Study Hours"]]           # Input
y = df["Predicted Output"]        # Output

# 3. Train the model
model = LinearRegression()
model.fit(X, y)

# 4. Save the model as .pkl
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl successfully!")
