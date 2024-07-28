import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv("Life-Expectancy-Data-Updated.csv")
print(df.head())

# Prepare the data
x = df[['Year', 'Under_five_deaths', 'Adult_mortality', 'BMI', 'Diphtheria', 'Polio', 'Thinness_ten_nineteen_years', 'Schooling']]
y = df["Life_expectancy"]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the model
rnf = RandomForestRegressor(random_state=30)
rnf.fit(x_train, y_train)

pickle.dump(rnf, open('model.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))