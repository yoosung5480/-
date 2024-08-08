import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class AdalineSGD(object):

    """
    확률적 선형 뉴런 분석기

    매개변수
        eta: float
            0.0 ~ 0.1 사이의 학습률
        n_iter: int 
            훈련 데이터셋 반복횟수
        shuffle : bool
            True로 설정되면 같은 반복이 일어나지 않도록, 에포크마다 훈련 데이터를 섞는다.
        random_stata : int
            가중치 무작위 초기화를 위한 난수 생성 시드번호

    속성
        w_ : 1차원 배열
            학습된 가중치
        cost : list
            모든 샘플의 에포크마다 누적된 평균 비용함수 제곱합
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle = True , random_state=None):
        self.eta = eta # 학습률
        self.n_iter = n_iter # 데이터셋 반복 횟수
        self.w_initialized = False # 가중치 초기화 여부
        self.shuffle = shuffle # 훈련 데이터 섞을지 여부
        self.random_state = random_state # 가중치 무작위 초기화를 위한 난수 생성기의 시드번호
    
    def fit(self, X, y):
    # param 
    # X : shape(n_samples, n_features)
    #   n개의 샘플과 n개의 특성으로 이뤄진 훈련 데이터
    # y : shape(1, n_samples) 
    #   타깃
        self._initialize_weights(X.shape[1]) 
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        # 가중치를 초기화 하지 않고 훈련 데이터를 학습한다.
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1 :
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else :
            self._update_weights(X, y)
        return self
            
    def _shuffle(self, X, y):
        # 훈련 데이터를 섞는다.
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        # 랜덤한 작은 수로 가중치를 초기화한다.
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        # 아달린 학습규칙에 따른 가중치 업데이트
        output = self.activation(self.net_input(xi)) # y* 값 (예측값 계산) 
        error = (target - output)                    # y - y* 실제값과 예측값 비교
        self.w_[1:] += self.eta * xi.dot(error)      # 학습률 * xi.dot(y - y*) xi는 Nx1크기의 n개의 특성을 갖는 1차원 배열 샘플 데이터일것이다. error의 값은 실수값일 것이다. 그런데 왜 두 값을 내적계산할까? -> error값이 실수값이 아니라 배열값인가?
        self.w_[0] += self.eta + error               # 가중치 첫값은 상수로써 바로 업데이트
        cost = 0.5 * error**2                        # 비용함수 : 1/2(y - w^t*x)**2
        return cost

    # y = x 꼴의 간단한 선형 활성화 함수
    def activation(self, X):
        return X

    # z = w^t * x 계산
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