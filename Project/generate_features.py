def generate_features(train_df, test_df):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Drop label column and concatenate
    df_all = pd.concat([train_df.drop(columns=["label"]), test_df], axis=0)

    # One-hot encode all object (categorical) columns
    df_all = pd.get_dummies(df_all)

    # Align features (ensure consistent columns across train/test)
    X = df_all.iloc[:len(train_df)]
    test = df_all.iloc[len(train_df):]

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    test = imputer.transform(test)

    # Scale for models that need it
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    test = scaler.transform(test)

    # Convert target to binary
    y = train_df["label"]

    return X, y, test