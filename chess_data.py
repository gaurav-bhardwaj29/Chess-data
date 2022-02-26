# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("games.csv")
df1 = df.copy()
df = pd.DataFrame(df)
df.drop(df.columns[[0, 2, 3, 8, 10, 12, 13, 15]], axis=1, inplace=True)  #Dropping attributes that will not be included for classification task

#  Data visualisation
#  Displaying top 5 most played openings for players below 1200 ELO rating

beginner = df[(df['white_rating'] < 1200) & (df['black_rating'] < 1200)]['opening_name'].value_counts().head(5)
plt.figure(figsize=(12, 5))
plt.title('Top 5 most played openings (below 1200 ELO)', fontsize=28)
plt.xlabel('Opening name', fontsize=24)
plt.xticks(fontsize=8)
plt.ylabel('Count', fontsize=24)
plt.yticks(fontsize=10)
sns.barplot(x=beginner.index, y=beginner.values, palette='magma')
plt.tight_layout()
plt.show()

#  Displaying top 5 most played openings for players between 1200-1800 ELO rating
intermediate = df[(df['white_rating'].between(1200, 1800)) & (df['black_rating'].between(1200, 1800))]['opening_name'].value_counts().head(5)
plt.figure(figsize=(12, 5))
plt.title('Top 5 most played openings (between 1200 and 1600 ELO)', fontsize=28)
plt.xlabel('Opening name', fontsize=24)
plt.xticks(fontsize=8)
plt.ylabel('Count', fontsize=24)
plt.yticks(fontsize=10)
sns.barplot(x=intermediate.index, y=intermediate.values, palette='magma')
plt.tight_layout()
plt.show()

#  Displaying top 5 most played openings for players above 1800 ELO rating
advanced = df[(df['white_rating'] > 1800) & (df['black_rating'] > 1800)]['opening_name'].value_counts().head(5)
plt.figure(figsize=(12, 5))
plt.title('Top 5 most played openings (above 1600 ELO)', fontsize=30)
plt.xlabel('Opening name', fontsize=24)
plt.xticks(fontsize=8)
plt.ylabel('Count', fontsize=24)
plt.yticks(fontsize=10)
sns.barplot(x=advanced.index, y=advanced.values, palette='magma')
plt.tight_layout()
plt.show()

#  Analysing how the game ended
df1["victory_status"].astype('str')
resign = df1.loc[df1["victory_status"] == 'resign'].count()[0]
outoftime = df1.loc[df1["victory_status"] == 'outoftime'].count()[0]
mate = df1.loc[df1["victory_status"] == 'mate'].count()[0]
draw = df1.loc[df1["victory_status"] == 'draw'].count()[0]
weights = [resign, outoftime, mate, draw]
labels = ["Resign", "Out of Time", "Checkmate", "Draw"]
plt.title("Result distribustion of players", fontdict={'fontname': 'sans-serif', 'fontweight': 'bold', 'fontsize': 18})
plt.pie(weights, labels=labels, autopct='%.2f %%', pctdistance=0.8)

plt.savefig('C:/Users/user/PycharmProjects/chess data/chess_data.png', dpi=200)
plt.show()
# Some more visualisation
plt.title("Player rating status", fontdict={'fontname': 'sans-serif', 'fontweight': 'bold', 'fontsize': 18})
plt.xlabel("Black rating")
plt.ylabel("White rating")
plt.scatter(df1["black_rating"], df1["white_rating"], s=1)
plt.show()
'''The linear relationship in the above scatter plot indicates that majority of players 
   played open challenges rather than creating custom games '''
from sklearn.preprocessing import LabelEncoder  # Assigning rated and winner labels integer values
encoder = LabelEncoder()
df["winner_cat"] = encoder.fit_transform(df["winner"])
df["rated_cat"] = encoder.fit_transform(df["rated"])
df["victory_status_cat"] = encoder.fit_transform((df["victory_status"]))
df.drop(["winner", "victory_status", "rated"], axis=1, inplace=True)
print(df["winner_cat"].value_counts())
# Cleaning the data


def get_increment(address):
    """returns Increment in seconds"""
    return address.split("+")[1].strip(" ")


def get_timeformat(address):
    """returns Time format in minutes"""
    return address.split("+")[0].strip("")


df['Time_format'] = df['increment_code'].apply(lambda x: f"{get_timeformat(x)}")
df['Increment'] = df['increment_code'].apply(lambda x: f"{ get_increment(x)}")
df["Time_format"] = pd.to_numeric(df["Time_format"])
df["Increment"] = pd.to_numeric(df["Increment"])

df.drop(["opening_name"], axis=1, inplace=True)

df.drop("increment_code", axis=1, inplace=True)

print(df.describe())
df["Time_format"].hist(bins=[0, 1.1, 3.1, 5.1, 10.1, 15.1, 20.1, 30.1, 60.1, 120.1], histtype='stepfilled')
plt.xticks([1, 3, 5, 10, 15, 20, 30, 60, 120])

plt.xlabel("Time Format")
plt.ylabel("# Games")
plt.show()
# Data preprocessing


def standardize(val):
    """Standardizes number of turns"""
    p = (val - val.mean())/val.std()
    return p


df["turns"] = standardize(df["turns"])

#  Passing only numeric labels

X = df.loc[:, ("turns", "victory_status_cat", "rated_cat")]
y = df.loc[:, "winner_cat"]
#  Trying out a bunch of Classification models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#  Implementing Decision Tree Classifier on training data
from sklearn.tree import DecisionTreeClassifier
tree_cls = DecisionTreeClassifier()
tree_cls.fit(X_train, y_train)
#  Implementing Decision Tree Classifier on test data
y_test_cls = tree_cls.predict(X_test)

from sklearn.metrics import accuracy_score
acc_dt = accuracy_score(y_test, y_test_cls)
#  Implementing K-Neighbours Classifier on training data
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train, y_train)
#  Implementing K-Neighbours Classifier on test data
y_test_knn = knn_model.predict(X_test)
acc_knn = accuracy_score(y_test, y_test_knn)
#  Implementing SGD Classifier on training data
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
#  Implementing SGD Classifier on test data
y_test_sgd = sgd_clf.predict(X_test)
acc_sgd = accuracy_score(y_test, y_test_sgd)
print("SDGC Classifier score: ", acc_sgd)
#  Implementing Logistic Regression Classifier on training data
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
#  Implementing Logistic Regression on test data
y_test_log = log_clf.predict(X_test)

#  Implementing Voting classifier to get the votes among various classifiers
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('dt', tree_cls), ('log', log_clf),
                                          ('knn', knn_model)], voting='hard')  # SDGC Classifier is not included due to its poor performance
voting_clf.fit(X_train, y_train)
for clf in (tree_cls, log_clf, knn_model,  voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#  Implementing Ensemble with Bagging classifier
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(tree_cls, n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
acc_bag = accuracy_score(y_test, y_pred)
print("Bagging Score= ", acc_bag)




