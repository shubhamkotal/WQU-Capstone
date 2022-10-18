Part 1: (Code 1)

The basic framework of the workflow is to use the OHLC (open-high-low-close) data and create multiple useful parameters i.e., feature engineering that can help to predict the next day's direction of the market based on the historical data.

After Importing required libraries. 

1.	Data collection: Downloading data from NSEPY for the symbol NIFTY. Where start date is (2010,1,1) and end date is (2022,8,27)

2.	Feature Extraction & Engineering

Created many required features from the OHLC data. Such as nifty_close_next_day, nifty_open_next_day, nifty_day_gain, nifty_day_chng

3.	Data cleaning: Dropping few redundant columns and formatting dates as per required.

4.	 A/B Testing (Train & Test Split) : Creating data sets to train and validate the model results

5.	Model Building: Creating two leading models in machine learning today, one based on bagging method and one on boosting Method.

a.	Random Forest Model
b.	XG Boost Model

The models been created with added variable refinement process.


6.	Model Results: Checking the model result. Using RMSE Score to evaluate model performance. The Current RMSE score is close to 1.




Part 2: (Code 2) 

For extra confirmation to predict the next day's direction of the market, we will be using the NLP models (Bert) to check the sentiment of the market based on recent trending news of the market. The sentiment prediction will be giving an extra edge for understanding the market strength and direction. Combining the above two decisions will be predicting the next day's direction.

       
Using a pre trained model FinBERT for analyzing the sentiments.

FinBERT is a pre-trained natural language processing (NLP) model that analyses the sentiment of financial writing. It is created by fine-tuning the BERT language model for financial sentiment categorization by further training it in the finance domain using a large financial corpus. 

1.	Importing required libraries and downloading required pre trained model files.
2.	Defining function required for cleaning the text data and further adding extra characters used in the operation of pre trained model operation.
3.	Giving labels to the classes
4.	Calling the pre-trained model
5.	Sample testing
6.	Results:


Example:

1.	Reliance trades will go high
Prediction: Neutral 
Score: 14%

2.	Reliance planning to expand jio networks.
Prediction: Positive
Score: 84%


Based on the above example, the testing results shows clever predictions been done. 
The pre trained model is been trained well and is able to classify the matter well. 
More try and checks were been done, before completely accepting the model.



Conclusion: 

As per the above explanation we were able to get the satisfied results. More refinements will be added going forward. 

