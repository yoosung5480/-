import numpy as np

class Perceptron(object):

    """
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
        errors_ : list
            에포크마다 누적된 오류 분포
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
        self.errors_ = []

        for _ in range(self.n_iter): 
            errors = 0
            for xi, target in zip(X, y): ## xi는 배열형태의 자료, target은 값일것이다.
                update = self.eta * (target - self.predict(xi)) 
                self.w_[1:] += update * xi # 두번째 인자부터는 일반적인 가중치이다. 이는 데이터의 입력에 따라 업데이트 돼야한다.
                self.w_[0] += update # w[0] 는 편향에 해당되는 가중치이다. 이는 xi 특성 (데이터의 입력)과 무관하게 업데이트 돼야한다. 
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    


    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] # 배열 X, W[1:]의 내적계산 후 편향값 더하기
    

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1) # 위의 인풋값에 대한 출력값을 매개로 해서, 만약 값이 0보다 크다면 1, 아니면 -1 출력