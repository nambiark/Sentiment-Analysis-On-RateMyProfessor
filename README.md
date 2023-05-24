# Sentiment-Analysis-On-RateMyProfessor
This project aims to generate sentiments from reviews extracted from the Rate My Professor website, with the goal of developing a consolidated view that benefits both students and professors. The objective is to understand the overall sentiment from multiple reviews, enhancing the learning experience in a comprehensive manner. Additionally, topic modeling techniques are employed to generate a summary that best describes each professor.

To achieve this, several models like Logistic Regression, Random Forest, KNN, and RNN, were trained using labeled data from sources such as IMDB and Amazon reviews. The models were evaluated using F1-score and then tested on unlabelled Rate My Professor reviews. The evaluation was also performed against Vader Sentiments.

## Code Flow:

-Vader Sentiment Analysis on Rate My Professor, IMDB, and Amazon reviews.
-Logistic Regression model trained and tested on IMDB and Amazon sentiments, with accuracy and F1-score obtained.
-The trained Logistic Regression model used to predict sentiments on Rate My Professor reviews.
-Random Forest model trained and tested on IMDB and Amazon sentiments, with accuracy and F1-score obtained.
-The trained Random Forest model used to predict sentiments on Rate My Professor reviews.
-KNN model trained and tested on IMDB and Amazon sentiments, with accuracy and F1-score obtained.
-The trained KNN model used to predict sentiments on Rate My Professor reviews.
-RNN model trained and tested on IMDB and Amazon sentiments, with accuracy and F1-score obtained.
-The trained RNN model used to predict sentiments on Rate My Professor reviews.


## Instructions to Run the Code:

1.Use Google Colab.
2.Upload all the required data files, including RateMyProfessor_Sample data, IMDB Dataset.csv, and amazon_cells_labelled.txt.
3.Run the code line by line.

preProcessing(text): This function takes the dataframe column comments as input and generates processed text as output. For example, the input is data['comments'], and the output is a string.

split_prep_data(): This function takes no input, sets the labeled data as X_train and y_train, and converts it into vectors using CountVectorizer. It returns the vectorized text, the vectorizer object, and the scaler object. For example, the output is vectorizer, scaler, X_train, y_train, X_test, and y_test.

testing_RMF(vectorizer, scaler, df): This function takes the vectorizer, scaler from the previous function, and the Rate My Professor dataframe. It splits the dataframe into test data and returns X_test and y_test after vectorizing and scaling.

get_f1(y_true, y_pred): This function takes the actual and predicted values of sentiments and computes the F1-score, as the Keras metric does not have a built-in F1-score metric. For example, the input is y_true, y_pred, and the output is the F1-score.

get_precision(y_true, y_pred): This function takes the actual and predicted values of sentiments and computes the precision, as the Keras metric does not have a built-in precision metric. For example, the input is y_true, y_pred, and the output is the precision value.

get_recall(y_true, y_pred): This function takes the actual and predicted values of sentiments and computes the recall, as the Keras metric does not have a built-in recall metric. For example, the input is y_true, y_pred, and the output is the recall value.
