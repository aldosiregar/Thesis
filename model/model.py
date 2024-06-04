import numpy as np
import pandas as pd
from sklearn import datasets, neighbors
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

class model:
    def __init__(self):
        self.df = pd.read_csv("Airlines.csv")

        #label encoding in init class
        airline_rep = self.df["Airline"].unique()
        airportfrom_rep = self.df["AirportFrom"].unique()
        airportto_rep = self.df["AirportTo"].unique()

        self.airline_keys = {}
        self.airportfrom_keys = {}
        self.airportto_keys = {}

        self.airline_keys = self.make_keys(airline_rep, self.airline_keys)
        self.airportfrom_keys = self.make_keys(airportfrom_rep, self.airportfrom_keys)
        self.airportto_keys = self.make_keys(airportto_rep, self.airportto_keys)

        self.df["Airline"].replace(self.airline_keys, inplace=True)
        self.df["AirportFrom"].replace(self.airportfrom_keys, inplace=True)
        self.df["AirportTo"].replace(self.airportto_keys, inplace=True)

        #drop kolom id
        self.df.drop(["id"], axis=1, inplace=True)

        #pemisahan fitur dan label
        self.x = self.df.drop("Delay", axis=1)
        self.y = self.df["Delay"]

        #feature selection dengan chi2
        self.chi_square_selection()

        #implement k-means in time and airline column
        #time column
        self.km_time = KMeans(
            n_clusters=3, init="random",
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )

        self.km_time.fit(self.x[:, 3].reshape(-1,1))
        self.x[:, 3] = self.km_time.predict(self.x[:, 3].reshape(-1,1))

        #airline column
        self.km_airline = KMeans(
            n_clusters=3, init="random",
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )

        self.km_airline.fit(self.x[:, 0].reshape(-1,1))
        self.x[:,0] = self.km_airline.predict(self.x[:, 0].reshape(-1,1))

        #normalization
        self.normalization()

        #split train data and test data
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25)

        #train model
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=9)
        self.clf.fit(x_train, y_train)

        #get model performance
        self.accuracy = accuracy_score(y_test, self.clf.predict(x_test))
        self.f1_scorer = f1_score(y_test, self.clf.predict(x_test))
        self.cross_val_scorer = cross_val_score(self.clf, self.x, self.y, cv=10)
        self.cross_val_mean = np.mean(self.cross_val_scorer)
        self.train_data_size = len(x_train)
        self.test_data_size = len(x_test)

    #make keys for label encoding
    def make_keys(self, keys, dictionary):
        for i in range(0, len(keys)):
            dictionary.update({keys[i] : i})
        return dictionary
    
    #chi square function
    def chi_square_selection(self):
        model = SelectKBest(chi2, k=5)
        self.x = model.fit_transform(self.x,self.y)
    
    #normalization function
    def normalization(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.x)
        self.x = self.scaler.transform(self.x)

    #get model performance
    def get_model_performance(self):
        return self.accuracy, self.f1_scorer, self.cross_val_mean
    
    def get_data_size(self):
        return self.train_data_size, self.test_data_size

    #prediction function
    def predict(self, data):
        x = data
        airline = x[0,0]
        airportto = x[0, 2]

        #translate airline and airportto data in form into label
        x[0, 0] = self.airline_keys[airline]
        x[0, 2] = self.airportto_keys[airportto]

        #implement kmeans
        x[0, 0] = self.km_airline.predict(np.reshape(x[0, 0], (1,-1)))
        x[0, 3] = self.km_time.predict(np.reshape(x[0, 3], (1,-1)))

        #implement normalization
        x = self.scaler.transform(x)

        return self.clf.predict(x)