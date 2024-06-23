import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score


from sklearn.tree import DecisionTreeClassifier


def main():

    # Sample training data (5 samples)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 1, 0, 1, 0])

    # Sample test data
    X_test = np.array([[2, 2], [3, 3], [6, 7]])

    # Create base models
    log_clf = LogisticRegression()
    tree_clf = DecisionTreeClassifier()
    svc_clf = SVC(probability=True)

    # Create an ensemble model using voting
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('dt', tree_clf), ('svc', svc_clf)],
        voting='hard'  # Use 'soft' for averaging predicted probabilities
    )

    # Train ensemble model
    voting_clf.fit(X_train, y_train)

    # Make predictions on the training data
    train_pred = voting_clf.predict(X_train)
    print("Predictions on training data:", train_pred)

    # Make predictions on the test data
    test_pred = voting_clf.predict(X_test)
    print("Predictions on test data:", test_pred)


if __name__ == "__main__":
    main()
