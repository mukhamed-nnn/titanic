import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
    train_mask = df['Survived'].notnull()

    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)

    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'Unknown')
    df['Deck'] = df['Deck'].replace({'T': 'Unknown', 'G': 'F'})

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_dict = {
        'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer', 'Dr': 'Officer', 'Rev': 'Officer',
        'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Sir': 'Royalty', 'Countess': 'Royalty', 'Lady': 'Royalty', 'Dona': 'Royalty',
        'Mme': 'Mrs', 'Ms': 'Miss', 'Mlle': 'Miss', 'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master'
    }
    df['Title'] = df['Title'].map(title_dict)

    df['Is_Mother'] = 0
    df.loc[(df['Age'] > 18) & (df['Sex'] == 'female') & (df['Parch'] > 0), 'Is_Mother'] = 1

    df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    df['Group_ID'] = df['LastName'] + '_' + df['Ticket']
    df['Family_Survival'] = 0.5
    for grp, group_df in df.groupby('Group_ID'):
        if len(group_df) > 1:
            for ind, row in group_df.iterrows():
                other_members = group_df.drop(ind)
                max_s = other_members['Survived'].max()
                min_s = other_members['Survived'].min()
                if max_s == 1.0:
                    df.loc[ind, 'Family_Survival'] = 1.0
                elif min_s == 0.0:
                    df.loc[ind, 'Family_Survival'] = 0.0
    
    df['Age'] = df['Age'].fillna(df[train_mask].groupby(['Sex', 'Pclass', 'Title'])['Age'].transform('mean'))
    df['Age'] = df['Age'].fillna(df[train_mask]['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df[train_mask]['Embarked'].mode()[0])

    df['Age_Class'] = df['Age'] * df['Pclass']

    fare_mask_train = (df[train_mask]['Fare'].notna()) & (df[train_mask]['Sex'] == 'male') & (df[train_mask]['Pclass'] == 3) & (df[train_mask]['Title'] == 'Mr')
    mean_fare_train = df[train_mask].loc[fare_mask_train, 'Fare'].mean()
    df['Fare'] = df['Fare'].fillna(mean_fare_train)

    ticket_counts = df['Ticket'].value_counts()
    df['Ticket_Frequency'] = df['Ticket'].map(ticket_counts)
    df['Fare_Per_Person'] = df['Fare'] / df['Ticket_Frequency']

    df['Sex'] = (df['Sex'] == 'female').astype(int)

    df.drop(columns=['Name', 'LastName', 'Group_ID', 'Ticket', 'Cabin'], inplace=True)

    train_cat = df[train_mask].copy()
    test_cat = df[~train_mask].copy().drop(columns=['Survived'])

    df = pd.get_dummies(df, columns=['Embarked', 'Title', 'Deck'], prefix=['Emb', 'Title', 'Deck'])

    


