# Disaster Response Pipeline Project

### Summary

Here we build a ETL and a ML pipeline to classify a text and get to know if it is related to some disasters. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files in the repository

The ETL pipeline is scripted in data/process_data.py while the ML pipeline in models/train_classifier.py. All codes related to the front end is under folder app. 

Here is an overview of the folder structure:
```
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # ETL pipeline
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py  # ML pipeline
|- classifier.pkl # saved model, it is too big and not saved in this repo
README.md
```


