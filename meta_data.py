import pandas as pd
import numpy as np
import torch
import gensim
from gensim.models import Word2Vec

def user_metadata():
    cols = ["id","age","gender","occupation","zip_code"]
    occupations = []
    user = pd.read_csv("/home/ducva/PythonProjects/dlrm/recommender_pytorch-master/Data/u.user", delimiter='|',
                                          names=cols, usecols=[0, 1, 2, 3],
                                          header=None, index_col=None, engine='c', encoding='latin-1')
    min_age = user.age.min()
    max_age = user.age.max()
    #print(min_age)
    #print(max_age)
    gender = {'M':1, 'F':0}
    user['age'] = user['age'].apply(lambda x: (x - min_age) / (max_age - min_age))
    user['gender'] = user['gender'].apply(lambda x: gender[x])
    user = pd.get_dummies(user, columns=['occupation'])
    #print(user.info())
    #print(user.head(1))
    #user.to_csv("/home/ducva/PythonProjects/dlrm/recommender_pytorch-master/Data/u.user", index=None)
    user_md = []
    for i in range(1, user.shape[0]+1):
        print(len(user.loc[user.id == i, ['age', 'gender', 'occupation_administrator', 'occupation_artist',
                                               'occupation_doctor', 'occupation_educator', 'occupation_engineer',
                                               'occupation_entertainment', 'occupation_executive',
                                               'occupation_healthcare', 'occupation_homemaker', 'occupation_lawyer',
                                               'occupation_librarian', 'occupation_marketing', 'occupation_none',
                                               'occupation_other', 'occupation_programmer', 'occupation_retired',
                                               'occupation_salesman', 'occupation_scientist', 'occupation_student',
                                               'occupation_technician', 'occupation_writer']].values.flatten().tolist()))
        user_md.append(str(user.loc[user.id == i, ['age', 'gender', 'occupation_administrator', 'occupation_artist',
                                               'occupation_doctor', 'occupation_educator', 'occupation_engineer',
                                               'occupation_entertainment', 'occupation_executive',
                                               'occupation_healthcare', 'occupation_homemaker', 'occupation_lawyer',
                                               'occupation_librarian', 'occupation_marketing', 'occupation_none',
                                               'occupation_other', 'occupation_programmer', 'occupation_retired',
                                               'occupation_salesman', 'occupation_scientist', 'occupation_student',
                                               'occupation_technician', 'occupation_writer']].values.flatten().tolist()))
        break

    print(len(user_md)) # 23 categories
    print(len(user_md))
    #u_vector = pd.DataFrame(user_md, columns=['vector'])
    #u_vector.to_csv("/home/ducva/PythonProjects/dlrm/recommender_pytorch-master/Data/u.vector", index=None)


def occupation_embedding():
    occupation = """administrator
artist
doctor
educator
engineer
entertainment
executive
healthcare
homemaker
lawyer
librarian
marketing
none
other
programmer
retired
salesman
scientist
student
technician
writer"""
    col_name = """occupation_administrator
occupation_artist
occupation_doctor
occupation_educator
occupation_engineer
occupation_entertainment
occupation_executive
occupation_healthcare
occupation_homemaker
occupation_lawyer
occupation_librarian
occupation_marketing
occupation_none
occupation_other
occupation_programmer
occupation_retired
occupation_salesman
occupation_scientist
occupation_student
occupation_technician
occupation_writer"""
    occupations = col_name.lower().split('\n')
    return occupations


if __name__=="__main__":
    user_metadata()
    #print(occupation_embedding())