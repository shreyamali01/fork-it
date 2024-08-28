# Task 3
We are using `llama3.1-8b` and `llama3.1-70b` models to do this task. Since the query is too large, it exceeded the token limit for most of the models. These two models are used to provide a better generalized result
## Question 1
We have used the 561-featured dataset to test and train the LLMs. After creating a csv file of the dataset along with the labels, we tested one example of every activity.<br>

Zero Shot Learning is the ability of a model to make predictions on data which was never seen by the model as part of the training, and is based on the existing data in the LLM model.<br>
Since there is no data for the model to train, the accuracy is generally lower than Few Shot Learning.
An advantage of Zero Shot Learning is that it can make predictions without any labeled data. This is useful in cases where obtaining labeled samples of every class is not possible.<br>
Zero Shot Learning was implemented on both the models.<br>
`llama3.1-8b` was able to generate the prediction of only one of the given test samples, which was predicted incorrectly. However, `llama3.1-70b` was able to predict one of the activity correctly, making the accuracy of Zero Shot Learning 16.67%.

Few Shot Learning is the ability of a model to make predictions based on limited labeled samples for each class.<br>
Based on the concept, Few Shot Learning usually performs better than Zero Shot Learning since knowing a few samples increases the accuracy of the model.
Few Shot Learning was implemented by giving two random example data of each activity considering the query token limit.<br>
`llama3.1-8b` was unable to generate any predictions, but `llama3.1-70b` model showed improvement. It was able to correctly predict two out of the six test samples given, making the accuracy of Few Shot Learning 33.33%.

## Question 2
The whole featured dataset containing 561 features and more than 7000 datapoints was used to train the Decision Tree, while 2 data samples of every class was used in Few Shot Learning. The metrics were observed as follows:<br>
Decision Tree Model: `Accuracy: 0.83, Precision: 0.75, Recall: 0.83`<br>
Few Shot Learning: `Accuracy: 0.33, Precision: 0.17, Recall: 0.33`<br>
The metrics clearly indicate the Decision Tree being better at accurately predicting the output. This is because the Decision Tree model is trained on a much larger dataset, making it capable of higher precision and accuracy, while the Few Shot Learning model had a very limited sample data. This causes the Few Shot Learning method to be much less accurate that its counterpart.

## Question 3
There are many limitations to Zero-Shot Learning and Few-Shot Learning in the context of classifying human activities based on featurized accelerometer data.<br>
1. Large Query Size: Since the featured dataset contains 561 features, the query often exceeds the query token limit. This limits the number of samples given in Few Shot Learning. Most of the models were unable to process such large queries, even in Zero Shot Learning, where the only data given was for testing.
2. Poor Generalization: Since human activites are prone to differences, the generalization from such a small dataset is not accurate enough.
3. Low Training Data: The large size of the features limits the number of data samples for the Few Shot Learning to be trained upon.

## Question 4
We tested the same data samples, but the query was changed to classify the data in WALKING, WALKING_UPSTAIRS and WALKING_DOWNSTAIRS. This means the model does not know that the activites STANDING, SITTING and LAYING could be possible options too. These activities were classified wrongly as one of the activities given to the model.<br>
This implies that when the model is faced with a data sample that corresponds to an activity not known to the model, it classifies it to the closest match.

## Question 5
We tested random data on the Zero Shot Learning and Few Shot Learning model. The results were the random data being classified in one of the given activities. The reason is same as in the previous question. When the model cannot find any close relation to any of the sample data (in case of Few Shot Learning), it chooses the closest match to be the correct class.