from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plots as p


def load_iris_data() -> dict:
    results = {}

    iris = load_iris()
    X, Y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    results["iris"] = iris
    results["train_set"] = (X_train, y_train)
    results["test_set"] = (X_test, y_test)

    return results


def ensemble_models(train_set, test_set):
    # Create base models
    log_clf = LogisticRegression()
    tree_clf = DecisionTreeClassifier()
    svm_clf = SVC(probability=True)

    X_train, y_train = train_set
    X_test, y_test = test_set

    # Fit base models
    log_clf.fit(X_train, y_train)
    tree_clf.fit(X_train, y_train)
    svm_clf.fit(X_train, y_train)

    # Create an ensemble model using voting
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('dt', tree_clf), ('svc', svm_clf)],
        voting='hard'  # Use 'soft' for averaging predicted probabilities
    )

    # Train ensemble model
    voting_clf.fit(X_train, y_train)

    # Make individual predictions
    log_pred = log_clf.predict(X_test)
    tree_pred = tree_clf.predict(X_test)
    svm_pred = svm_clf.predict(X_test)

    # Make final predictions
    voting_pred = voting_clf.predict(X_test)

    # Print individual predictions
    print("Logistic Regression predictions: ", log_pred)
    print("Decision Tree predictions: ", tree_pred)
    print("SVM predictions: ", svm_pred)

    # Evaluate model
    accuracy = accuracy_score(y_test, voting_pred)
    print(f'Ensemble model accuracy: {accuracy:.2f}')


if __name__ == "__main__":
    results = load_iris_data()

    train_set = results["train_set"]
    test_set = results["test_set"]
    iris = results["iris"]

    ensemble_models(train_set, test_set)

    p.scatter_plot(iris)
