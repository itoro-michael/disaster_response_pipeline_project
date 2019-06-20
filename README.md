# Disaster Response Pipeline Project

In an emergency created by a natural disaster alot of messages are generated by its victims. 
The content of such messages usually reflect the different needs of the senders. The needs can include
request for ambulance, fire service, or for food. These messages are transmitted through various media 
such as sms and social media. The overwhelming volume of these messages make it impractical for human processing.
A machine learning model is needed to effectively classify the different messages generated and channel them to 
the appropriate services for processing.

This project is an emergency message classifier designed to group a message into one or more of 36 categories. 
Some of the categories are Request, Aid Related, Medical Help, Food, Shelter, and Clothing. The MultiOutputClassifier 
and the AdaBoostClassifier from the sklearn package in Python are combined and trained to produce a multi-label 
classification model. The model accuracy on test set is at 93%, with precision of 0.76, and 0.98 in recall. The data used 
for training the model is provided by Figure8.

## Files in the project besides the README:

1. data/process_data.py: The Python module that loads the disaster_messages.csv and disaster_categories.csv datasets,
						 cleans them and saves the transformed data in the DisasterResponse.db database file.
2. data/disaster_messages.csv: The csv file containing messages obtained during natural disasters,  
							   which are used for model training.
3. data/disaster_categories.csv: The csv file that contains the labels for each message.
4. data/DisasterResponse.db: The database file containing the cleaned dataset used for model training.
5. models/train_classifier.py: The Python module used to define the classifier model, train, evaluate and save
							   the model to the classifier.pkl file.
6. models/classifier.pkl: The pickle file for saved model.

### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
