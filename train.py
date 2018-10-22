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

output_file="evaluated_csv.csv"
def KNN_model(train_data):
    # K-NN model
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(train_data)
    with open("./data/knn_model.pickle", "wb") as f:
        pickle.dump(model_knn, f)
        f.close()
    return model_knn

def test_knn(model,test_data):
    print("test_knn()")
    n_books = test_data.shape[0]
    print(n_books)
    user_ids=dict(test_data.iloc[1,:].astype('int64'))
    user_ids=list(user_ids.keys())
    n_user_ids=len(user_ids)
    print(n_user_ids)
    # n_books=100
    
    books1=[]

    fields=['user_id']
    for j in range(n_user_ids):
        distances, indices = model.kneighbors(test_data.iloc[j, :].reshape(1, -1), n_neighbors = n_books)
        dict_data=dict({'user_id':user_ids[j]})
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for the book {0} and used id {1} :\n'.format(test_data.index[j] ,user_ids[j]))
            else:
                fields.append(test_data.index[indices.flatten()[i]].replace(':',' '))
                dict_data.update({str(test_data.index[indices.flatten()[i]]).replace(':',' ') :distances.flatten()[i]})
                print('{0}: {1}, with distance of {2}:'.format(i, test_data.index[indices.flatten()[i]], distances.flatten()[i]))
        books1.append(dict_data)

    write_csv_file(output_file,fields,books1)

    print("Done.......")

# load pickle data
def load_data(pickle_name):
    with open(pickle_name, "rb") as f:
        return pickle.load(f)  

def write_csv_file(csv_file_path,fields,data):
    print("write_csv_file")
    with open(csv_file_path, 'w' ,newline='',encoding='utf-8') as csvFile:
        fields = fields
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
        csvFile.close()

if __name__=='__main__':
    pickle_name1="./data/data_set1.pickle"
    pickle_name2="./data/data_set2.pickle"

    train_data,test_data,data=load_data(pickle_name1)
    
    knn_model=KNN_model(train_data)
    test_knn(knn_model,test_data)

    