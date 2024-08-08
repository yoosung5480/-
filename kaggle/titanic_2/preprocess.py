import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Preprocessor():
    def __init__(self, df):
        self.df = df
    
    def _honorifics_classify(df_honorific):
        honorific_list = ['Mr', 'Miss', 'Mrs' ,'Master']
        return df_honorific if df_honorific in honorific_list else 'else'
    
    # 'Name'행에서 명칭만 추출해낸 후, 'Mr', 'Miss', 'Mrs' ,'Master'외에는 else로 인코딩한다.
    # 이 과정이 끝나고나면 결측치 처리후 바로 원핫 인코딩 가능해진다.
    def _extract_honorific(self):
        name_df = self.df['Name'].str.split(pat='[,.]', n=2, expand = True)
        name_df.columns = ['family_name', 'honorific', 'name']
        for column in name_df.columns:
            name_df[column] = name_df[column].str.strip()
        self.df = pd.concat([self.df, name_df['honorific']], axis=1)
        self.df['honorific'] = self.df['honorific'].apply(self._honorifics_classify)
        return
    
    # self.droped_df 결측치를 모두 제거하고, 원-핫 인코딩
    # drop_list를 통해서 실수가 아닌 행들과 원-핫 인코딩하기 전의 모든 항의 드랍할것이다.
    # 모든결측치가 제거되고 모든 데이터가 인코딩된 self.droped_df를 객체가 가지게 한다.
    def _one_hot_encoding(self, df):
        self.droped_df = self.df.dropna(axis=0)
        ohe = OneHotEncoder()
        ohe_targetlists = ['Sex', 'Embarked', 'honorific']
        drop_list = ['PassengerId','Age','Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'honorific']
        ohe_arr = ohe.fit_transform(self.droped_df[ohe_targetlists]).toarray()
        ohe_arr_feature_names = ohe.get_feature_names_out(ohe_targetlists)
        # concat은 인덱스가 맞지 않으면 행이 추가되는 오류가 생긴다.
        # 원-핫 인코딩한 배열을 data frame으로 만들때, 그럼으로 반드시 index=원본df.index 를 통해서 인덱스를 통일시켜야한다.
        ohe_arr_df = pd.DataFrame(ohe_arr, columns=ohe_arr_feature_names, index=self.droped_df.index)
        self.droped_df = pd.concat([self.droped_df, ohe_arr_df], axis=1)
        df_train = df_train.drop(drop_list, axis=1)
        return

    # self.df의 drop_list의 행이름 외의 항에 결측치가 있다면, self.droped_df의 값을 평균값으로 메꿀것이다.
    def _na_process_(self):
        ## 현재 상태
        # self.df 크기가 예를들어 864행의 데이터 프레임
        # self.droped_df 결측치가 있는 행을 제거해서 크기가 예를들어 714정도로 작아진 데이터 프레임. 게다가 원핫 인코딩 진행됨.
        # 
        return

    def preprocess(self):
        self._extract_honorific()
        self._one_hot_encoding()

        return
    
# 전처리 클래쓰
# Name에서 명칭만 추출 후 ['Mr', 'Miss', 'Mrs' ,'Master']외에는 else로 범주화
# ['Sex', 'Embarked', 'honorific']에 대해서 원-핫 인코딩 수행
# 처음 데이터에서 ['PassengerId','Age','Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'honorific'] 제거


# 특정행의 결측치를 평균값으로 채우는 함수 만들기.
