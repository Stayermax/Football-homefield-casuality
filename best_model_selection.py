from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
# from xgboost import XGBClassifier, XGBRegressor
from metric_learn import NCA, MLKR
import pickle
import numpy as np

# list of scorings: https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
# list of objectives : https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
# Bayesian optimization

class NCAKNN():
    def __init__(self, n_neighbors=-1, n_components=None, max_iter=100,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        self._knn = KNeighborsClassifier(n_neighbors=n_neighbors if n_neighbors > 0 else 5, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
                 p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self._metric_learner = NCA(init='auto', n_components=n_components, max_iter=max_iter, verbose=1)
        self._feature_importances_ = None
        self._find_k = False if n_neighbors > 0 else True

    @staticmethod
    def list_of_ks(samples):
        n = max(int(samples * 0.05), 20)
        sieve = [True] * n
        for i in range(3, int(n ** 0.5) + 1, 2):
            if sieve[i]:
                sieve[i * i::2 * i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
        primes = [1 ,2] + [i for i in range(3, n, 2) if sieve[i]]
        ks = []
        i, jump = 0, 0
        while i < len(primes):
            if i % 4 == 0:
                jump += 1
            ks.append(primes[i])
            i += jump
        return ks

    @property
    def find_k(self):
        return self._find_k

    @property
    def knn(self):
        return self._knn

    def transform(self, X):
        return self._metric_learner.transform(X)

    def fit(self, X, y, **kwargs):
        self._metric_learner.fit(X, y)
        self._feature_importances_ = np.sum(self._metric_learner.components_.T, axis=1)
        self._feature_importances_ = self._feature_importances_ / np.sum(self._feature_importances_)
        fitted_X = self.transform(X)
        return self.knn.fit(fitted_X, y)

    def predict(self, X):
        fitted_X = self.transform(X)
        return self.knn.predict(fitted_X)

    def predict_proba(self, X):
        fitted_X = self.transform(X)
        return self.knn.predict_proba(fitted_X)

    @property
    def feature_importances_(self):
        return self._feature_importances_


class MLKRKNN():
    def __init__(self, n_neighbors=5, n_components=None, max_iter=1000, weights='uniform',
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        self._knn = KNeighborsRegressor(n_neighbors=n_neighbors if n_neighbors > 0 else 5, weights=weights,
                                        algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric,
                                        metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self._metric_learner = MLKR(init='auto', n_components=n_components, max_iter=max_iter, verbose=1)
        self._feature_importances_ = None
        self._find_k = False if n_neighbors > 0 else True

    @staticmethod
    def list_of_ks(samples):
        return NCAKNN.list_of_ks(samples)

    @property
    def find_k(self):
        return self._find_k

    @property
    def knn(self):
        return self._knn

    def transform(self, X):
        return self._metric_learner.transform(X)

    def fit(self, X, y, **kwargs):
        self._metric_learner.fit(X, y)
        self._feature_importances_ = np.sum(self._metric_learner.components_.T, axis=1)
        self._feature_importances_ = self._feature_importances_ / np.sum(self._feature_importances_)
        fitted_X = self.transform(X)
        return self.knn.fit(fitted_X, y)

    def predict(self, X):
        fitted_X = self.transform(X)
        return self.knn.predict(fitted_X)

    @property
    def feature_importances_(self):
        return self._feature_importances_


class BestModel:
    def __init__(self, model='randomforest', init_points=3, n_iter=2, **init_kwargs):
        if not isinstance(model, str):
            raise Exception("model should be a string")
        self._model_str = model.lower().replace(' ', '')
        if self._model_str in ('best', 'bestmodel', 'bestclassifier', 'classifier'):
            self._init_function = None
            self._optimize_function = None
            self._model_str = 'classifier'
        elif self._model_str in ('bestregressor', 'regressor'):
            self._init_function = None
            self._optimize_function = None
            self._model_str = 'regressor'
        # elif self._model_str in ('xgboost', 'xgbclass', 'xgboostclassifier', 'xgbclassifier'):
        #     self._init_function = self._init_XGBClassifier
        #     self._optimize_function = self._optimize_XGBClassifier
        #     self._model_str = 'xgboost'
        elif self._model_str in ('randomforest', 'rfc', 'randomforestclassifier'):
            self._init_function = self._init_RandomForestClassifier
            self._optimize_function = self._optimize_RandomForestClassifier
            self._model_str = 'rfc'
        # elif self._model_str in ('xgbreg', 'xgboostregressor', 'xgbregressor'):
        #     self._init_function = self._init_XGBRegressor
        #     self._optimize_function = self._optimize_XGBRegressor
        #     self._model_str = 'xgbreg'
        elif self._model_str in ('rfr', 'randomforestregressor'):
            self._init_function = self._init_RandomForestRegressor
            self._optimize_function = self._optimize_RandomForestRegressor
            self._model_str = 'rfr'
        elif self._model_str in ('lr', 'logisticregression'):
            self._init_function = self._init_LogisticRegression
            self._optimize_function = self._optimize_LogisticRegression
            self._model_str = 'lr'
        elif self._model_str in ('linearregression', 'lasso'):
            self._init_function = self._init_Lasso
            self._optimize_function = self._optimize_Lasso
            self._model_str = 'lasso'
        elif self._model_str in ('knn', 'knnclassifier', 'knnclass', 'nca', 'ncaknn'):
            self._init_function = self._init_NCAKNN
            self._optimize_function = self._optimize_NCAKNN
            self._model_str = 'ncaknn'
        elif self._model_str in ('knnregressor', 'knnreg', 'mlkr', 'mlkrknn'):
            self._init_function = self._init_MLKRKNN
            self._optimize_function = self._optimize_MLKRKNN
            self._model_str = 'mlkrknn'
        else:
            raise Exception("Model should be one of the following: 'xgboost', 'rcf', 'rfr', 'lr', 'lasso', 'ridge', 'ncaknn', 'mlkrknn'")
        self._init_kwargs = init_kwargs
        self._fitted_model = None
        self._feature_importances_ = None
        self._init_points = init_points
        self._n_iter = n_iter

    def _bayesian_optimization(self, cv_function, parameters):
        gp_params = {"alpha": 1e-5, 'init_points': self._init_points, 'n_iter': self._n_iter, }
        bo = BayesianOptimization(cv_function, parameters)
        bo.maximize(**gp_params)
        return bo.max

    def get_init_kwargs(self):
        if self._init_function is not None:
            return {k: v for k, v in self._init_kwargs.items() if k in self._init_function.__code__.co_varnames}
        else:
            return {}


    # -------------- init & optimize functions -------------- #

    # ----- rfc -----

    @staticmethod
    def _init_RandomForestClassifier(params, **kwargs):
        return RandomForestClassifier(
            n_estimators=int(max(params['n_estimators'], 1)),
            max_depth=int(max(params['max_depth'], 1)),
            min_samples_split=int(max(params['min_samples_split'], 2)),
            min_samples_leaf=int(max(params['min_samples_leaf'], 2)),
            n_jobs=-1,
            random_state=42,
            class_weight='balanced')

    @staticmethod
    def _optimize_RandomForestClassifier(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
            return cross_val_score(BestModel._init_RandomForestClassifier(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"n_estimators": (10, 500),
                      "max_depth": (5, 100),
                      "min_samples_split": (2, 100),
                      "min_samples_leaf": (2, 50)}
        return cv_function, parameters

    # ----- rfr -----

    @staticmethod
    def _init_RandomForestRegressor(params, **kwargs):
        return RandomForestRegressor(
            n_estimators=int(max(params['n_estimators'], 1)),
            max_depth=int(max(params['max_depth'], 1)),
            min_samples_split=int(max(params['min_samples_split'], 2)),
            min_samples_leaf=int(max(params['min_samples_leaf'], 2)),
            n_jobs=-1,
            random_state=42)

    @staticmethod
    def _optimize_RandomForestRegressor(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, **kwargs):
        def cv_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
            return cross_val_score(BestModel._init_RandomForestRegressor(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"n_estimators": (10, 500),
                      "max_depth": (5, 100),
                      "min_samples_split": (2, 100),
                      "min_samples_leaf": (2, 50)}
        return cv_function, parameters

    # ----- xgboost classifier -----

    @staticmethod
    def _init_XGBClassifier(params, objective='binary:logistic', **kwargs):
        return XGBClassifier(
            objective=objective,
            learning_rate=max(params['eta'], 0),
            gamma=max(params['gamma'], 0),
            max_depth=int(max(params['max_depth'], 1)),
            n_estimators=int(max(params['n_estimators'], 1)),
            min_child_weight=int(max(params['min_child_weight'], 1)),
            seed=42,
            nthread=-1)

    @staticmethod
    def _optimize_XGBClassifier(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(eta, gamma, max_depth, n_estimators, min_child_weight):
            params = {'eta': eta, 'gamma': gamma, 'max_depth': max_depth, 'n_estimators': n_estimators,
                      'min_child_weight': min_child_weight}
            return cross_val_score(BestModel._init_XGBClassifier(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"eta": (0.001, 0.4),
                      "gamma": (0, 15),
                      "max_depth": (1, 100),
                      "n_estimators": (1, 500),
                      "min_child_weight": (1, 20)}
        return cv_function, parameters

    # ----- xgboost regressor ----- #

    @staticmethod
    def _init_XGBRegressor(params, objective='reg:squarederror', **kwargs):
        return XGBRegressor(
            objective=objective,
            learning_rate=max(params['eta'], 0),
            gamma=max(params['gamma'], 0),
            max_depth=int(max(params['max_depth'], 1)),
            n_estimators=int(max(params['n_estimators'], 1)),
            min_child_weight=int(max(params['min_child_weight'], 1)),
            seed=42,
            nthread=-1)

    @staticmethod
    def _optimize_XGBRegressor(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, **kwargs):
        def cv_function(eta, gamma, max_depth, n_estimators, min_child_weight):
            params = {'eta': eta, 'gamma': gamma, 'max_depth': max_depth, 'n_estimators': n_estimators,
                      'min_child_weight': min_child_weight}
            return cross_val_score(BestModel._init_XGBRegressor(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"eta": (0.001, 0.4),
                      "gamma": (0, 15),
                      "max_depth": (1, 100),
                      "n_estimators": (1, 500),
                      "min_child_weight": (1, 20)}
        return cv_function, parameters

    # ----- lasso ----- #

    @staticmethod
    def _init_Lasso(params, **kwargs):
        if params['alpha'] < 0.25:
            return LinearRegression(n_jobs=-1)
        else:
            return Lasso(alpha=max(params['alpha'], 0.25))

    @staticmethod
    def _optimize_Lasso(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, **kwargs):
        def cv_function(alpha):
            params = {'alpha': alpha}
            return cross_val_score(BestModel._init_Lasso(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"alpha": (0.0, 10)}
        return cv_function, parameters

    # ----- lr ----- #

    @staticmethod
    def _init_LogisticRegression(params, **kwargs):
        return LogisticRegression(C=max(params['C'], 0.0), max_iter=2000 ,solver='liblinear')

    @staticmethod
    def _optimize_LogisticRegression(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(C):
            params = {'C': C}
            return cross_val_score(BestModel._init_LogisticRegression(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"C": (0.0, 1000)}
        return cv_function, parameters

    # ----- ncaknn -----

    @staticmethod
    def _init_NCAKNN(params, n_neighbors=-1, max_iter=100, **kwargs):
        print(n_neighbors, max_iter, kwargs)
        return NCAKNN(
            max_iter=max_iter,
            n_neighbors=n_neighbors,
            n_components=int(max(params['n_components'], 2)),
            n_jobs=-1)

    @staticmethod
    def _optimize_NCAKNN(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(n_components):
            params = {'n_components': n_components}
            ncaknn = BestModel._init_NCAKNN(params, **kwargs)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
            ncaknn.fit(X_train, y_train)
            X_test_fitted = ncaknn.transform(X_test)
            if ncaknn.find_k:
                scores = []
                for k in NCAKNN.list_of_ks(len(X_test)):
                    ncaknn.knn.n_neighbors = k
                    scores.append(cross_val_score(ncaknn.knn, X=X_test_fitted, y=y_test, cv=cv_splits,
                                       scoring=scoring, n_jobs=n_jobs).mean())
                return np.max(scores)
            else:
                return cross_val_score(ncaknn.knn, X=X_test_fitted, y=y_test, cv=cv_splits,
                                       scoring=scoring, n_jobs=n_jobs).mean()
        parameters = {"n_components": (2, min(X.shape[1], 500))}
        return cv_function, parameters

    # ----- mlkrknn -----

    @staticmethod
    def _init_MLKRKNN(params, n_neighbors=-1, max_iter=1000, **kwargs):
        return MLKRKNN(
            max_iter=max_iter,
            n_neighbors=n_neighbors,
            n_components=int(max(params['n_components'], 2)),
            n_jobs=-1)

    @staticmethod
    def _optimize_MLKRKNN(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, **kwargs):
        def cv_function(n_components):
            params = {'n_components': n_components}
            mlkrknn = BestModel._init_MLKRKNN(params, **kwargs)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
            mlkrknn.fit(X_train, y_train)
            X_test_fitted = mlkrknn.transform(X_test)
            if mlkrknn.find_k:
                scores = []
                for k in MLKRKNN.list_of_ks(len(X_test)):
                    mlkrknn.knn.n_neighbors = k
                    scores.append(cross_val_score(mlkrknn.knn, X=X_test_fitted, y=y_test, cv=cv_splits,
                                                  scoring=scoring, n_jobs=n_jobs).mean())
                return np.max(scores)
            else:
                return cross_val_score(mlkrknn.knn, X=X_test_fitted, y=y_test, cv=cv_splits,
                                       scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"n_components": (2, min(X.shape[1], 500))}
        return cv_function, parameters

    # -------------- sklearn API functions -------------- #

    def fit(self, X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1):
        if self._model_str in ('classifier', 'regressor'):
            # models_to_check = ('rfc', 'lr') if self._model_str == 'classifier' else ('lasso', 'rfr')
            models_to_check = ('rfc', 'lr') if self._model_str == 'classifier' else ['rfr']
            best_model, best_score, best_params = None, None, None
            for model_str in models_to_check:
                if self._model_str == 'regressor':
                    if(model_str == 'lasso'):
                        self._init_points = 75
                        self._n_iter = 50
                        self._init_kwargs = {'max_iter': 100, 'n_neighbors': -1}
                    else:
                        self._init_points = 3
                        self._n_iter = 3
                        self._init_kwargs = {'max_iter': 6, 'n_neighbors': -1}
                print(f'------------------ working on {model_str} ------------------')
                model = BestModel(model_str, self._init_points, self._n_iter)
                cv_function, parameters = model._optimize_function(X, y, cv_splits, scoring, n_jobs, **self.get_init_kwargs())
                best_solution = model._bayesian_optimization(cv_function, parameters)
                params, score = best_solution["params"], best_solution["target"]
                print(f'\tResults for {model_str}:\n\t\tbest params={params}\n\t\tbest score={score}')
                if best_score is None or score > best_score:
                    best_model, best_score, best_params = model, score, params
            best_model._fit(X, y, cv_splits, scoring, n_jobs, best_params)
            self.__dict__.update(best_model.__dict__)
            return self._fitted_model
        else:
            return self._fit(X, y, cv_splits, scoring, n_jobs)

    def _fit(self, X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, params=None):
        if params is None:
            cv_function, parameters = self._optimize_function(X, y, cv_splits, scoring, n_jobs, **self.get_init_kwargs())
            best_solution = self._bayesian_optimization(cv_function, parameters)
            params = best_solution["params"]
        model = self._init_function(params, **self.get_init_kwargs())
        model.fit(X, y)
        self._fitted_model = model
        if self._model_str in ('lr', 'lasso'):
            self.feature_importances_ = self._fitted_model.coef_
        else:
            self.feature_importances_ = self._fitted_model.feature_importances_
        return self._fitted_model

    def predict(self, X):
        if self._fitted_model is not None:
            return self._fitted_model.predict(X)
        else:
            raise Exception('Model should be fitted before prediction')

    def fit_predict(self, X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1):
        self.fit(X, y, cv_splits, scoring, n_jobs)
        return self.predict(X)

    def transform(self, X):
        if self._model_str in ('nca', 'mlkr'):
            if self._fitted_model is not None:
                return self._fitted_model.transform(X)
            else:
                raise Exception('Model should be fitted before prediction')
        else:
            raise Exception('Only nca or mlkr models have transform method')

    def predict_proba(self, X):
        if self._fitted_model is not None:
            return self._fitted_model.predict_proba(X)
        else:
            raise Exception('Model should be fitted before prediction')

    @property
    def feature_importances_(self):
        if self._feature_importances_ is not None:
            return self._feature_importances_
        else:
            raise Exception('model should be fitted before feature_importances_')

    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances_ = value

    def save_model(self, fname):
        with open(fname, 'wb') as file:
            pickle.dump(self._fitted_model, file)

    def load_model(self, fname):
        with open(fname, 'rb') as file:
            self._fitted_model = pickle.load(file)
