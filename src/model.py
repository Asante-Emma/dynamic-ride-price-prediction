import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def train_model(csv_data_path, target):
    # Load data into a pandas dataframe
    data = pd.read_csv(csv_data_path)

    # Select features
    features = ['Number_of_Riders', 'Number_of_Drivers', 'Location_Category', 
                'Customer_Loyalty_Status', 'Number_of_Past_Rides', 'Average_Ratings', 
                'Time_of_Booking', 'Vehicle_Type', 'Expected_Ride_Duration']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Define preprocessing steps for numerical and categorical features
    numeric_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 
                        'Average_Ratings', 'Expected_Ride_Duration', 'Historical_Cost_of_Ride']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Fit the model
    model.fit(X_train, y_train)

    return model, X_test, y_test
