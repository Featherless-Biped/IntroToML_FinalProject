def generate_features(train_df, test_df, return_feature_names=False):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    df_all = pd.concat([train_df.drop(columns=["label", "ID"]), test_df.drop(columns=["ID"])], axis=0)

    # One-hot encode all object (categorical) columns
    df_all = pd.get_dummies(df_all)
    feature_names = df_all.columns.tolist()  # 🟢 Get column names here

    X = df_all.iloc[:len(train_df)]
    test = df_all.iloc[len(train_df):]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    test = imputer.transform(test)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    test = scaler.transform(test)

    y = train_df["label"]

    if return_feature_names:
        return X, y, test, feature_names
    else:
        return X, y, test
