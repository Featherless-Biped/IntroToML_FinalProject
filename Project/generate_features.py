import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generate_features(train_df, test_df, return_feature_names=False,
                      shuffle_train_id=False, shuffle_test_id=False,
                      inject_random_test_id=False,drop_id_completely=False):
    """
    Generates processed features for training and testing with controlled handling of ID.
    
    Parameters:
    - shuffle_train_id: If True, shuffles the ID column in train
    - shuffle_test_id: If True, shuffles the ID column in test
    - drop_id_completely: If True, removes ID from both train and test

    Returns:
    - X, y, test, feature_names (if return_feature_names=True)
    """

    # Clone input to avoid mutation
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Shuffle ID values if requested
    if shuffle_train_id:
        print("🌀 Shuffling ID in TRAIN")
        train_df["ID"] = np.random.permutation(train_df["ID"].values)
    if shuffle_test_id:
        print("🌀 Shuffling ID in TEST")
        test_df["ID"] = np.random.permutation(test_df["ID"].values)
    if inject_random_test_id:
        print("🎲 Injecting completely random ID values in TEST")
        test_df["ID"] = np.random.randint(1_000_000, 10_000_000, size=len(test_df))
    

    # Drop ID if requested
    if drop_id_completely:
        print("🧹 Dropping ID from both train and test")
        train_df = train_df.drop(columns=["ID"])
        test_df = test_df.drop(columns=["ID"])

    # Prepare features
    train_features = train_df.drop(columns=["label"])
    test_features = test_df.copy()

    # Concatenate and one-hot encode
    df_all = pd.concat([train_features, test_features], axis=0)
    df_all = pd.get_dummies(df_all)

    # Save feature names BEFORE splitting back
    feature_names = df_all.columns.tolist()

    # Split back into train and test
    X = df_all.iloc[:len(train_df)]
    test_processed = df_all.iloc[len(train_df):]

    # Impute and scale
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    test_processed = imputer.transform(test_processed)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    test_processed = scaler.transform(test_processed)

    # # Optional: Print ID samples for sanity check
    # if "ID" in train_df.columns and "ID" in test_df.columns:
    #     print("🔎 ID check:")
    #     print("Train ID example:", train_df["ID"].iloc[:3].values)
    #     print("Test ID example :", test_df["ID"].iloc[:3].values)
    #     print("Intersection:", set(train_df["ID"]).intersection(set(test_df["ID"])))
    # else:
    #     print("⚠️ ID column not present in one of the datasets.")


    y = train_df["label"]

    if return_feature_names:
        return X, y, test_processed, feature_names
    else:
        return X, y, test_processed
