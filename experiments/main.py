from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# TODO: Come up with a better name
# TODO: Add documentation comments
class MagicBox:
    def __init__(self, model, **params):
        self.model = model
        self.estimator = None
        self.prediction = None

        self.set_params(**params)

    def fit(self, X, y):
        self.estimator = self.model.fit(X, y)

    def predict(self, X):
        self.prediction = self.estimator.predict(X)
        return self.prediction

    def score(self, X, y):
        if self.prediction is None:
            self.predict(X)

        print(f'RMSE = {mean_squared_error(y, self.prediction, squared=False)}')
        print(f'R2 = {self.estimator.score(X, y)}')

    def evaluate(self):
        pass

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    magic_box = MagicBox(DecisionTreeRegressor(), **{"random_state": 1})
    magic_box.fit(X_train, y_train)
    magic_box.score(X_test, y_test)
