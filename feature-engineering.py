import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
def feature_engineering(df):
    # Handle missing values
    df.fillna(df.median(), inplace=True)

    # Create new features
    df['feature_sum'] = df.sum(axis=1)
    df['feature_mean'] = df.mean(axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Encode categorical features
    encoder = OneHotEncoder(sparse=False)
    categorical_features = df.select_dtypes(include=['object']).columns
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    # Combine the original dataframe with the encoded features
    df = pd.concat([df.drop(categorical_features, axis=1), encoded_df], axis=1)

    return df