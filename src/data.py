"""Загрузка исходных данных и построение признаков для Titanic."""
import pandas as pd


def load_raw_data(path_to_train: str, path_to_test: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path_to_train)
    test = pd.read_csv(path_to_test)
    return train, test


def preprocessing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Строит признаки, кодирует категории и разделяет обратно на train/test.

    Возвращает (train, test, train_cat, test_cat), где *_cat — версии
    с "сырыми" категориальными колонками (без One-Hot) для CatBoost.
    """
    train_mask = df["Survived"].notna()

    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notnull(x) else "Unknown")
    df["Deck"] = df["Deck"].replace({"T": "Unknown", "G": "F"})

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_dict = {
        "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer", "Rev": "Officer",
        "Jonkheer": "Royalty", "Don": "Royalty", "Sir": "Royalty", "Countess": "Royalty",
        "Lady": "Royalty", "Dona": "Royalty",
        "Mme": "Mrs", "Ms": "Miss", "Mlle": "Miss",
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    }
    df["Title"] = df["Title"].map(title_dict)

    df["Is_Mother"] = 0
    df.loc[(df["Age"] > 18) & (df["Sex"] == "female") & (df["Parch"] > 0), "Is_Mother"] = 1

    group_keys = ["Sex", "Pclass", "Title"]
    group_mean_age = df.loc[train_mask].groupby(group_keys)["Age"].mean()
    age_from_train_groups = pd.Series(df.set_index(group_keys).index.map(group_mean_age), index=df.index)
    df["Age"] = df["Age"].fillna(age_from_train_groups).fillna(df.loc[train_mask, "Age"].median())
    df["Age_Class"] = df["Age"] * df["Pclass"]

    df["Embarked"] = df["Embarked"].fillna(df.loc[train_mask, "Embarked"].mode()[0])

    fare_median = df.loc[
        train_mask & (df["Pclass"] == 3) & (df["Embarked"] == "S")
        & (df["Title"] == "Mr") & (df["SibSp"] + df["Parch"] == 0),
        "Fare",
    ].median()
    df.loc[df["Fare"].isna(), "Fare"] = fare_median

    ticket_counts = df["Ticket"].value_counts()
    df["Ticket_Frequency"] = df["Ticket"].map(ticket_counts)
    df["Fare_Per_Person"] = df["Fare"] / df["Ticket_Frequency"]

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Sex"] = (df["Sex"] == "female").astype(int)

    df = df.drop(columns=["PassengerId", "Ticket", "Name", "Cabin"])
    df_cat = df.copy()

    df = pd.get_dummies(df, columns=["Embarked", "Title", "Deck"])
    df = df.drop(columns=["Embarked_Q", "Title_Royalty", "Deck_Unknown"])

    train, test = df[train_mask].copy(), df[~train_mask].copy()
    train_cat, test_cat = df_cat[train_mask].copy(), df_cat[~train_mask].copy()
    test = test.drop(columns=["Survived"])
    test_cat = test_cat.drop(columns=["Survived"])

    return train, test, train_cat, test_cat