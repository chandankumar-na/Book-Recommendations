import pandas as pd 
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import csv
import os

from flask import Flask,abort,Request,Response,render_template,request,redirect,url_for
from flask import Flask
app = Flask(__name__)

# load pickle data
def load_data(pickle_name):
    if os.path.exists(pickle_name):
        with open(pickle_name, "rb") as f:
            return pickle.load(f)  

def load_model(model_name):
    if os.path.exists(model_name):
        with open(model_name, "rb") as f:
            return pickle.load(f)  


pickle_name1="./data/data_set1.pickle"
pickle_name2="./data/data_set2.pickle"
model_name='./data/knn_model.pickle'
_,test_dataset,_=load_data(pickle_name1)
knn_model=load_model(model_name)


def test_knn(model,test_data,n_books,u_id):
    print("test_knn()")

    user_ids=dict(test_data.iloc[1,:].astype('int64'))
    user_ids=list(user_ids.keys())

    n_user_ids=len(user_ids)
    u_id_index=user_ids.index(u_id)

    distances, indices = model.kneighbors(test_data.iloc[u_id_index, :].reshape(1, -1), n_neighbors = n_books+1)
    
    books1=[]
    scores=[]
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for the book {0} and used id {1} :\n'.format(test_data.index[u_id_index], user_ids[u_id_index]))
        else:   
            print('{0}: {1}, with distance of {2}:'.format(i, test_data.index[indices.flatten()[i]], distances.flatten()[i]))
            books1.append([str(test_data.index[indices.flatten()[i]]).strip(),distances.flatten()[i]])
            scores.append(distances.flatten()[i])
    print("Done.......")
    return books1 ,scores 




def get_user_ids(test_data):
    print("get_user_ids")
    user_ids=dict(test_data.iloc[1,:].astype('int64'))
    user_ids=list(user_ids.keys())
    return user_ids

def main():
    n_books=10
    u_id=387
    recommendations=test_knn(knn_model,test_dataset,n_books,u_id)
    print(recommendations)


@app.route('/')
def index():
    users_all=get_user_ids(test_dataset)
    return render_template("index.html", userIds=users_all)

@app.route('/recommendations' ,methods=['POST'])
def recommendations():
    print("recommendations()")
    users_all=get_user_ids(test_dataset)

    selected_user_id = int(request.form.get("selected_user_id"))
    num_books =int(request.form.get("n_books"))

    print(selected_user_id)
    print(num_books)


    books1 ,scores =test_knn(knn_model,test_dataset,num_books,selected_user_id)
    selected_user_id=[selected_user_id]

    return render_template("index.html",selected_user_id=selected_user_id , r_books=books1 , r_score= scores,userIds=users_all)


@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(port=5000,debug = True)