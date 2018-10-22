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

output='output.csv'
users_csv='Users.csv'
books_csv='Books.csv'
user_events_csv='UserEvents.csv'



#Readinng CSV file and filtering 
def read_csv(csv_file_path,columns):
    print('read_csv({})'.format(csv_file_path))
    df=pd.read_csv(csv_file_path, encoding='latin-1', usecols=columns)
    print("Original length {} and shape {}".format(len(df),df.shape))
    return df 
        


def filter_data():
    print("filter_data()")

    #reading Books details from csv file
    book_columns = ['ISBN', 'bookName', 'author', 'yearOfPublication', 'publisher']
    books_data=read_csv(books_csv,book_columns)

    books_data.yearOfPublication=pd.to_numeric(books_data.yearOfPublication,errors='coerce')
    books_data.loc[(books_data.yearOfPublication==0),'yearOfPublication']=np.NaN
    books_data.yearOfPublication.fillna(round(books_data.yearOfPublication.mean()),inplace=True)
    books_data.yearOfPublication=books_data.yearOfPublication.astype(np.int32)
 
   

    # Reading User details from csv file
    user_columns = ['user_id', 'location', 'age']   
    users_data=read_csv(users_csv,user_columns)


    # users_data = users_data.dropna(how='any',axis=0)

    users_data.loc[(users_data.age>90) | (users_data.age<5),'age']=np.NaN
    users_data.age=users_data.age.fillna(users_data.age.mean())
    users_data.age=users_data.age.astype(np.int32)

    #Reading user event details from csv file
    user_events_columns = ['user_id', 'ISBN', 'impression'] 
    user_events_data=read_csv(user_events_csv,user_events_columns)
    print("isbn unique", len(user_events_data['ISBN'].unique()))


    # user_events_pivot = user_events_data.pivot(index='user_id', columns='ISBN').impression


    #Combining(merge) user_events_data and books data based on  ISBN
    combine_books_user_events = pd.merge(user_events_data, books_data, on='ISBN')
    # print(combine_books_user_events)


    #Removing yearOfPublication, publisher and author from combined data
    droped_columns = ['yearOfPublication', 'publisher', 'author']
    combine_books_user_events = combine_books_user_events.drop(droped_columns, axis=1)

    # Dropping NaN data in "bookName" Column from combined data
    combine_books_user_events = combine_books_user_events.dropna(axis = 0, subset = ['bookName'])
    # print(combine_books_user_events)

    
    # grouping data based on the impression(i,e ->1.add cart, 2.checkout, 3.view, 4.like, 5.Dislike, 6.Intract) and counting  
    book_user_eventsCount = (combine_books_user_events.groupby(by = ['bookName'])['impression'].count().reset_index()
    .rename(columns = {'impression': 'totalImpressionCount'})[['bookName', 'totalImpressionCount']])

    # print(book_user_eventsCount)


    #Combining(merge) user_events_data and books data based on  ISBN
    user_events_with_total_user_events_Count = combine_books_user_events.merge(book_user_eventsCount, left_on = 'bookName', right_on = 'bookName', how = 'left')

    # print(user_events_with_total_user_events_Count)

    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    popularity_threshold = 5
    impression_popular_book = user_events_with_total_user_events_Count.query('totalImpressionCount >= @popularity_threshold')
    # print(impression_popular_book)



    #Combining(merge) user details based on user id  with already combined and filtered "user_events_data and books" i,e=>impression_popular_book
    combined_all_data = impression_popular_book.merge(users_data, left_on = 'user_id', right_on = 'user_id', how = 'left')
    
    #filling NaN where value is null in location column
    combined_all_data['location'] = combined_all_data['location'].fillna('')
    print("Combining all the 3 data ")
    # print(combined_all_data)

    #Filtering the data whre location has only canada and usa
    us_canada_user_impression = combined_all_data[combined_all_data['location'].str.contains("usa|canada ")]
    # us_canada_user_impression=combined_all_data
    # print(us_canada_user_impression)

    # us_canada_user_impression=us_canada_user_impression.drop('age', axis=1)
    # print(us_canada_user_impression)

    #Dropping duplicate who has same user id with same book name
    us_canada_user_impression = us_canada_user_impression.drop_duplicates(['user_id', 'bookName'])
    # print(us_canada_user_impression)


    us_canada_user_impression_pivot = us_canada_user_impression.pivot(index = 'bookName', columns = 'user_id', values = 'age').fillna(0)
    
    us_canada_user_impression_matrix = csr_matrix(us_canada_user_impression_pivot.values)
    with open("data_set2.pickle", "wb") as f:
        pickle.dump((us_canada_user_impression_matrix,us_canada_user_impression_pivot,us_canada_user_impression), f)
        f.close()
    return us_canada_user_impression_matrix,us_canada_user_impression_pivot,us_canada_user_impression

if __name__=='__main__':
    train_data,test_data,data=filter_data()
   
  

 




    



