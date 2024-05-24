from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def LogisticRegressionFunction(comment_train_tfidf, labels_train):

    param_distributions = {
        'C': uniform(loc=0.01, scale=10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200]
    }

    logistic_regression = LogisticRegression(multi_class='ovr')

    random_search = RandomizedSearchCV(logistic_regression, param_distributions=param_distributions,
                                       n_iter=10, scoring='balanced_accuracy',
                                       cv=3, verbose=2, random_state=42)

    random_search.fit(comment_train_tfidf, labels_train)

    print("Best parameters: ", random_search.best_params_)
    print("Best balanced accuracy score (CV): ", random_search.best_score_)

    return random_search.best_params_, random_search.best_score_