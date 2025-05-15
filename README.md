# Sentiment Analysis and Text Classification of Twitter Data
## Project Description
Sentiment Analysis is a Natural Language Processing task where the goal is to determine the emotional tone behind textual data. In this project, I will collect, preprocess, and classify social media posts from platform X and organize them into the following sentiment categories: Positive, Negative, and Neutral. 

## Run Instructions
1. Download the four Python files, main, data_loader, preprocessing, and sentiment_model, from the python_files folder
2. Download the two CSV files, twitter_training and twitter_validation, from the data folder
3.  Launch PyCharm CE and start a new project in a virtual environment using Python 3.13.4
4.  Install all the necessary libraries listed in the Dependencies section below
5.  Create a folder called data in the project and add the two CSV files to that folder
6.  Add all the Python files to the project and save each of them.
7.  Run main.py and wait until the program completes, at which point you will be provided with graphs analysing the model
   
Note: Once the model runs, the information from the model will be saved as roberta_gru_sentiment.pt, which can be utilized for further analysis

## Dependencies
- pandas
- scikit-learn
- numpy
- torch
- transformers
- nltk
- matplotlib
- regex
