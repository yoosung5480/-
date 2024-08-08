import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re


class Data:
    def __init__(self, data):
        self.data = data 

    def _extract_title(self, name):
        title = re.search(r'\b(\w+)\.',name)
        return title.group(1)
    
    def _encode_title(self, title):
        title_dict = {
            'Mr' : 1, 
            'Miss' : 2, 
            'Mrs' : 3,
            'Master' : 4, 
            'Dr' : 5, 
            'Rev' : 6
        }
        if title in title_dict.keys():
            return title_dict[title]
        else:
            return 0
        
    def _encode_Name(self):
        title = self.data['Name'].apply(self._extract_title)
        encoded_name = title.apply(self._encode_title)
        self.data['Name'] = encoded_name

    def _encode_Embarked(self):
        self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes

    def _encode_Sex(self):
        self.data['Sex'] = self.data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    def process_data(self):
        self._encode_Embarked()
        self._encode_Name()
        self._encode_Sex()
        y_data = self.data['Survived']
        x_data = self.data[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        return x_data, y_data

