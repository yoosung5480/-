import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class AdalineGD(object):

    """
    선형 뉴런 분석기

    매개변수
        eta: float
            0.0 ~ 0.1 사이의 학습률
        n_iter: int 
            훈련 데이터셋 반복횟수
        random_stata : int
            가중치 무작위 초기화를 위한 난수 생성 시드번호

    속성
        w_ : 1차원 배열
            학습된 가중치
        cost : list
            데이터셋 마다의 비용 제곱합
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
    # param 
    # X : shape(n_samples, n_features)
    #   n개의 샘플과 n개의 특성으로 이뤄진 훈련 데이터
    # y : shape(1, n_samples) 
    #   타깃

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # numpy의 randomstate 객체를 반환한다. 이를 이용하면 일관된 난수 시퀸스를 생성하게 된다.
        # 첫번째 원소는 편향값이다. 이 가중치들을 0으로 초기화 하게되면 가중치 벡터의 방향에는 영향을 주지못하고, 크기에만 영향을 미친다.
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    # y = x 꼴의 간단한 선형 활성화 함수
    def activation(self, X):
        return X

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] # 배열 X, W[1:]의 내적계산 후 편향값 더하기
    

    def predict(self, X):
        # 단위 계단함수를 사용하여 클래쓰 레이블은 만든다.
        return np.where(self.net_input(X) >= 0.0, 1, -1) # 위의 인풋값에 대한 출력값을 매개로 해서, 만약 값이 0보다 크다면 1, 아니면 -1 출력
    
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.2):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그리는 부분
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 각 클래스 별로 데이터 점 표시
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=cl, edgecolor='black')

    # 테스트 인덱스가 있을 경우, 테스트 데이터셋 특별 표시
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    edgecolor='black', alpha=1.0, linewidth=1,
                    marker='o', s=100, label='test set')

    plt.legend(loc='upper left')
    plt.show()