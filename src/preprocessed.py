import numpy as np

def preprocess_data(data):
  """
  Preprocesses data for use in the dynamic pricing model

  Args:
      data: A pandas DataFrame containing the ride data

  Returns:
      A preprocessed pandas DataFrame
  """
  # Identify numeric and categorical features
  numeric_features = data.select_dtypes(include=['float', 'int']).columns
  categorical_features = data.select_dtypes(include=['object']).columns

  # Handle missing values in numeric features
  data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())

  # Handle outliers in numeric features
  for feature in numeric_features:
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    data[feature] = np.clip(data[feature], lower_bound, upper_bound)

  # Handle missing values in categorical features (consider alternative methods)
  data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

  return data