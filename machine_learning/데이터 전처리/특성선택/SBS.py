from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size = 0.25, random_state=1):
        self.scoring = scoring
        self.estimator = estimator
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1] # 특성의 개수
        self.indices_ = tuple(range(dim)) #현재 선택된 특성의 인덱스 저장
        self.subsets_ = [self.indices_] # 각단계에서 선택된 특성의 조합 저장.
        score = self._calc_score(X_train, X_test, y_train, y_test, self.indices_)
        self.scores_ = [score]


        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, X_test, y_train, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]
        return self


    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, X_test, y_train, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
df_wine.columns = ['Class label', 
                   'Alcohol', 
                   'Malic acid', 
                   'Ash',
                   'Alcanlinity of Ash', 
                   'Magnesium',
                   'Total phenols',
                   'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 
                   'Hue',
                   '0D280/0D315 of diluted wines',
                   'proline']


def main():
    # 데이터 로드 및 열 이름 설정
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcanlinity of Ash', 'Magnesium',
                       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                       'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    # 데이터 준비
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # 표준화
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)  # fit_transform과 transform을 구분

    # 모델 생성 및 SBS 실행
    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    # 선택된 특성에 대한 성능 출력
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    k4 = list(sbs.subsets_[9])
    print(df_wine.columns[1:][k4])

if __name__ == "__main__":
    main()