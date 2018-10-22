# Book Recommendations Appliction

* **Overview**
  I developed an application using Machine learning to recommend the books to users based on reported users data .

* **Algorithm Used** 
    - Collaborative filtering [https://en.wikipedia.org/wiki/Collaborative_filtering]
    - K-NN (K- Nearest neighbor) [https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]


* **Programing language/Frameworks used**
    - Python
    - Flask

* **Steps to run the code :**
    I built two different type of data sets and trained with ML model .I used KNN algorithm to predict top most similar (likly) to  buy books. The appliaction can also be run on the flask server with simple UI. To run UI based application user has to open the browser and run the url [http://localhost:5000/]. User must select the User Id as well as number books to be recommend.It will show the top most Books and score in ascending order.

    - open the project source folder and run the below command on cmd
    
    - Python server.py (To run on flask server) [http://localhost:5000/]
    - python train.py  (To build M x N matrix csv file)



* **Note:** 
    - I have Also deployed the application into Heroku server. Please do check once (it will easy to test)
    click here  [https://chandan-recomendations.herokuapp.com/]
    - Please do find a ziped evaluated csv file



