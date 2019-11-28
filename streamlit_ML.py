import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

@st.cache
def loadData():
	df = pd.read_csv("data.csv")
	return df

# Basic preprocessing required for all the models.
def preprocessing(df):
	X = df.iloc[:,[0,3,5]].values
	y = df.iloc[:,-1].values

	#Encoding the data of X and y as it is Categorical
	le = LabelEncoder()
	y = le.fit_transform(y.flatten())

	#Splitting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state=0)
	return X_train,X_test,y_train,y_test,le

# Using Decision Tree for Classification
@st.cache(suppress_st_warning=True)
def decisionTree(X_train,X_test,y_train,y_test):
	#Model Training
	tree = DecisionTreeClassifier(max_leaf_nodes=3,random_state=0)
	tree.fit(X_train,y_train)
	y_pred = tree.predict(X_test)
	score = metrics.accuracy_score(y_test,y_pred)*100
	report = classification_report(y_test,y_pred)

	return score, report, tree

#Training Neural Network for Classification
@st.cache(suppress_st_warning=True)
def neuralNet(X_train,X_test,y_train,y_test):
	#Scaling the data
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	#Instantiate the Classifier and fit the model
	clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	score1 = metrics.accuracy_score(y_test,y_pred)*100
	report = classification_report(y_test,y_pred)

	return score1, report, clf

#Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train,X_test,y_train,y_test):
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test,y_pred)*100
	report = classification_report(y_test,y_pred)

	return score, report, clf


#Accepting user data for predicting member type
def accept_user_data():
	duration = st.text_input("Enter the Duration: ")
	start_station = st.text_input("Enter the start station number: ")
	end_station = st.text_input("Enter the end station number: ")
	user_prediction_data = np.array([duration,start_station,end_station]).reshape(1,-1)

	return user_prediction_data

@st.cache
def showMap():
	plotData = pd.read_csv("trip_data.csv")
	Data = pd.DataFrame()
	Data['lat'] = plotData['lat']
	Data['lon'] = plotData['lon']

	return Data

def main():
	st.title("Predicting the Trip History Data through different Machine Learning Algorithms.")
	data = loadData()
	X_train, X_test, y_train, y_test, le = preprocessing(data)

	#Inserting Check Box to display snippet of data
	if st.checkbox('Showing Raw Data'):
		st.subheader("Showing raw data :")
		st.write(data.head())

	#Section for prediction
	choose_model = st.sidebar.selectbox("Let's choose the model",["NONE","Decision Tree","Neural Network","K-Nearest Neighbours"])

	if(choose_model == "Decision Tree"):
		score, report, tree = decisionTree(X_train,X_test,y_train,y_test)
		st.text("The Decision Tree model has accuracy of: ")
		st.write(score,"%")
		st.text("Decision Tree Model report is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data()
				pred = tree.predict(user_prediction_data)
				st.write("The class predicted is: ",le.inverse_transform(pred))
		except:
			pass

	elif(choose_model == "Neural Network"):
		score, report, tree = decisionTree(X_train,X_test,y_train,y_test)
		st.text("The Neural Network model has accuracy of: ")
		st.write(score,"%")
		st.text("Neural Network Model report is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data()
				scaler = StandardScaler()
				scaler.fit(X_train)
				user_prediction_data = scaler.transform(user_prediction_data)
				pred = clf.predict(user_prediction_data)
				st.write("The class predicted is: ",le.inverse_transform(pred))
		except:
			pass

	elif(choose_model == "K-Nearest Neighbours"):
		score, report, tree = decisionTree(X_train,X_test,y_train,y_test)
		st.text("The K-Nearest Neighbours model has accuracy of: ")
		st.write(score,"%")
		st.text("K-Nearest Neighbours Model report is: ")
		st.write(report)

		try:
			if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
				user_prediction_data = accept_user_data()
				pred = clf.predict(user_prediction_data)
				st.write("The class predicted is :",le_inverse_transform(pred))
		except:
			pass

	# Visualization Section
	plotData = showMap()
	st.subheader("Bike Travel History data plotted-first few locations located near Washington DC")
	st.map(plotData, zoom = 14)


	choose_viz = st.sidebar.selectbox("Choose the Visualization",
		["NONE","Total number of vehicles from various Starting Points", "Total number of vehicles from various End Points",
		"Count of each Member Type"])
	
	if(choose_viz == "Total number of vehicles from various Starting Points"):
		fig = px.histogram(data['Start station'], x ='Start station')
		st.plotly_chart(fig)
	elif(choose_viz == "Total number of vehicles from various End Points"):
		fig = px.histogram(data['End station'], x ='End station')
		st.plotly_chart(fig)
	elif(choose_viz == "Count of each Member Type"):
		fig = px.histogram(data['Member type'], x ='Member type')
		st.plotly_chart(fig)

	# plt.hist(data['Member type'], bins=5)
	# st.pyplot()

if __name__ == "__main__":
	main()


