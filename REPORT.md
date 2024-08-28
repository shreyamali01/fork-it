# Assignment 1

## Task 1 : Exploratory Data Analysis (EDA)

1. **Plot of One Sample Data from Each Class**

    ![Plot of One Sample Data from Each Class](Task_1/T1Plot1.png)

    It can be infered from the graphs that the amplitudes of acceleration of static activities (Laying, Standing, Sitting) is much smaller than that of the dynamic activities (Walking, Walking Upstairs, Walking Downstairs).

    The model can well differentiate / classify activities as static and dynamic only. But because of less complexity of the model, it can't differentiate the 6 activities much accurately. Since the data is time-sensored data, use of Neural Network Algorithms will be able to classify each activity.

2. **Density Plot of Linear Acceleration for Different Activities**

    ![Plot](Task_1/T1Question2.png)

    ### STATIC ACTIVITIES
      * The narrow acceleration range classifies static activities like laying, standing and sitting, as the waveform also   shows a near flatline for such activities. The concentrated density signifies minimal variation in acceleration, reflecting the relatively stationary nature of these activities.
    
    ### DYNAMIC ACTIVITIES
      * The broad acceleration range classifies dynamic activities like walking, walking downstairs and walking upstairs, as the waveform also shows a high change in acceleration for such activities. The reduced prominence of density peaks implies a greater variability in acceleration values, which is indicative of movement and changes in activity.
    
    Based on the linear acceleration properties of the activities, this plot helps in the distinction between static and dynamic activities. It is feasible to determine whether an activity is largely stationary or involves movement by examining the density peaks and the acceleration value distribution. With the help of this data, models that can categorize activities according to their patterns of acceleration will be developed, which would benefit applications like activity recognition and health monitoring.

3. **Visualizing the data using PCA**

      * Scatter Plot to visualize different classes of activities (PCA on Total Acceleration)

        ![Plot](Task_1/T1Q3.png)

      * Scatter Plot to visualize different classes of activities (PCA on features created by using TSFEL)

        ![Plot](Task_1/T1Q3b.png)
      
      * Scatter Plot to visualize different classes of activities (PCA on features in the provided dataset)

        ![Plot](Task_1/T1Q3c.png)

    Yes, the scatter plot has become more comprehensive and understandable by using TSFEL library which featurizes the data and gives better results as compared to PCA. The accuracy increased on using the TSFEL library. The PCA plot had more overlapping points making it difficult to analyse the plot unlike the TSFEL plot where it was scattered giving better results.

4. **Question 4**

    Please refer to `Task_1.ipynb`


## Task 2 : Decision Trees for Human Activity Recognition

1. **Use of Sklearn Library to Train Decision Trees**

    * **Results for decision tree model trained using the raw accelerometer data:**

      The code for training a decision tree model using the raw accelerometer data - `'sec1.ipynb`
    
        - Accuracy: 0.61
        - Precision: 0.60
        - Recall: 0.61
        
        - Confusion Matrix:
          ```
          [[34368     0     0     0     0     0]
           [    0 23461  5141   385   896  1541]
           [    0  5893 23824   483     0  3848]
           [   16  2301  2800 13051  3369 10207]
           [   23  1527  1698  9328  4788  9516]
           [   26  1241  2575  9532  1752 15018]]
          ```

        Here, from the confusion matrix, we can observe that the model predicts correctly for class 1. However, it isn't very accurate when it comes to the rest of the classes. To mitigate this inconsistency, we tried oversampling the minority class by applying SMOTE (Synthetic Minority Over-Sampling Technique). This implementation yields the following results:
    
    * **Results after applying SMOTE:**
    
        - Accuracy: 0.62
        - Precision: 0.62
        - Recall: 0.62
        
        - Confusion Matrix:
          ```
          [[34368     0     0     0     0     0]
           [    0 25329  4001   548     0  1546]
           [    0  6748 23028   410     0  3862]
           [   16  4143  2579 10958  6374  7674]
           [   23  3434  1549  5784 10525  5565]
           [   26  2989  2365  8063  3431 13270]]
          ```
    
        To improve the results even further, we tried undersampling the majority and the results were as follows:
    
    * **Results after applying Undersampling:**
    
        - Accuracy: 0.63
        - Precision: 0.62
        - Recall: 0.63
        
        - Confusion Matrix:
          ```
          [[34368     0     0     0     0     0]
           [    0 25323  4011   441   103  1546]
           [    0  6666 23064   351   110  3857]
           [   16  2987  2611 10912  7508  7710]
           [   23  2338  1558  5768 11566  5627]
           [   26  1430  2379  8024  4952 13333]]
          ```

        Thus, it can be observed that even after implementing oversampling and undersampling, significant changes in the result are not observed. Yet, undersampling provides slightly better results compared to oversampling.

    * **Results for decision tree model trained using the features obtained by TSFEL:**

      The code for training a decision tree model using the features obtained by TSFEL - `sec2.ipynb`
    
        - Accuracy: 0.89
        - Precision: 0.90
        - Recall: 0.89
        
        - Confusion Matrix:
          ```
          [[9 0 0 0 0 0]
           [1 7 1 0 0 0]
           [1 1 7 0 0 0]
           [0 0 0 7 2 0]
           [0 0 0 0 9 0]
           [0 0 0 0 0 9]]
          ```
        
        Oversampling and undersampling yielded the same results for the decision tree model trained with this data.
    
    * **Results for decision tree model trained using the features provided in the dataset:**

      The code for training a decision tree model using the features provided in the database - `sec3.ipynb`
    
        - Accuracy: 0.84
        - Precision: 0.84
        - Recall: 0.84
        
        - Confusion Matrix:
          ```
          [[537   0   0   0   0   0]
           [  0 400  91   0   0   0]
           [  0 107 425   0   0   0]
           [  0   0   0 462  18  16]
           [  0   0   0  31 276 113]
           [  0   0   0  51  35 385]]
          ```

        Oversampling and undersampling yielded the same results for the decision tree model trained with this data.

    * **Comparison Between the 3 Models**

        The model trained on raw accelerometer data shows poor performance across all metrics compared to the other two models. The model trained on the features provided in the dataset shows better performance than the raw data model. However, the model trained on the TSFEL features shows the best performance with an accuracy of 89% and precision of 90%. From the confusion matrix, we observe that it works well for most of the classes. The reason for better performance is that TSFEL extracts relevant features, making it easier for the model to distinguish between the classes/activities.

2. **Train Decision Trees with Varying Depths (2-8) Using All the Above 3 Methods**
    
    ### Plot of accuracy vs depth of the trees for the models

    - Model trained on Raw Accelerometer Data 

    ![Accuracy vs. Depth](Task_2/accuracy_vs_depth_model1.png)

    - Model trained using features extracted from TSFEL

    ![Accuracy vs. Depth](Task_2/accuracy_vs_depth_model2.png)

    - Model trained on features provided in the dataset

    ![Accuracy vs. Depth](/Task_2/accuracy_vs_depth_model3.png)

3. **Bad Activities and Participants**

    Upon analyzing the confusion matrices for model 1, model 2 and model 3, we notice that all the models fit perfectly for class 1, and model 3 fits perfectly for class 1, class 5 and class 6. Furthermore, it can be also observed that Model 1 has the most
    misclassifications followed by model 3 and model 2 respectively. The reason for this miscalculations is class imbalance. We tried to resolve the class imbalance by trying to oversample the minority classes and undersample the majority classes, however it did not make much difference to the accuracy values, precision values and confusion matrices. 

    We can also determine the best and worst performing activities/classes by F1 score measure. 



## Task 3 : Prompt Engineering for Large Language Models (LLMs)

We are using `llama3.1-8b` and `llama3.1-70b` models to do this task. Since the query is too large, it exceeded the token limit for most of the models. These two models are used to provide a better generalized result
### Question 1
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

### Question 2
The whole featured dataset containing 561 features and more than 7000 datapoints was used to train the Decision Tree, while 2 data samples of every class was used in Few Shot Learning. The metrics were observed as follows:<br>
Decision Tree Model: `Accuracy: 0.83, Precision: 0.75, Recall: 0.83`<br>
Few Shot Learning: `Accuracy: 0.33, Precision: 0.17, Recall: 0.33`<br>
The metrics clearly indicate the Decision Tree being better at accurately predicting the output. This is because the Decision Tree model is trained on a much larger dataset, making it capable of higher precision and accuracy, while the Few Shot Learning model had a very limited sample data. This causes the Few Shot Learning method to be much less accurate that its counterpart.

**Question 3** 

There are many limitations to Zero-Shot Learning and Few-Shot Learning in the context of classifying human activities based on featurized accelerometer data.<br>
1. Large Query Size: Since the featured dataset contains 561 features, the query often exceeds the query token limit. This limits the number of samples given in Few Shot Learning. Most of the models were unable to process such large queries, even in Zero Shot Learning, where the only data given was for testing.
2. Poor Generalization: Since human activites are prone to differences, the generalization from such a small dataset is not accurate enough.
3. Low Training Data: The large size of the features limits the number of data samples for the Few Shot Learning to be trained upon.

**Question 4**
We tested the same data samples, but the query was changed to classify the data in WALKING, WALKING_UPSTAIRS and WALKING_DOWNSTAIRS. This means the model does not know that the activites STANDING, SITTING and LAYING could be possible options too. These activities were classified wrongly as one of the activities given to the model.<br>
This implies that when the model is faced with a data sample that corresponds to an activity not known to the model, it classifies it to the closest match.

**Question 5**
We tested random data on the Zero Shot Learning and Few Shot Learning model. The results were the random data being classified in one of the given activities. The reason is same as in the previous question. When the model cannot find any close relation to any of the sample data (in case of Few Shot Learning), it chooses the closest match to be the correct class.

## Task 4 : Data Collection in the Wild

1. **Result of Decision Tree Model model trained on the raw accelerometer data on our data is as follows**

    Accuracy: 0.24
    Precision: 0.19
    Recall: 0.24
    Confusion Matrix:
      [[5162    0    0    0    0    0]
      [5738   13    2    2   77  113]
      [4705   22    0    0   21   47]
      [3246  104    3 2270  109  228]
      [3067  112    1 1602   75  163]
      [2984  132    2 1847   71  194]]
    
    Here we observe that the accuracy is very low and from the confusion matrix we can observe that the data shows high imbalance. The model works well for the class 1 whereas it isn't able to predict other activities efficiently. To improve on the accuracy we can try to preprocess the data.

2. **Result of Decision Tree Model model trained on our data is as follows**

    Accuracy: 0.51
    Precision: 0.40
    Recall: 0.51
    Confusion Matrix:
      [[ 703  209  106    2    0    0]
      [  92  808  186   85    0    0]
      [  72  233  603   50    0    0]
      [   0   21   26 1138    0    0]
      [   0   10   29  976    0    0]
      [   0   22   37 1015    0    0]]
    
    Let's normalize the data now, 

    Accuracy: 0.17
    Precision: 0.03
    Recall: 0.17
    Confusion Matrix:
      [[   0    0    0 1020    0    0]
      [   0    0    0 1171    0    0]
      [   0    0    0  958    0    0]
      [   3   62   51 1069    0    0]
      [   8   92   48  867    0    0]
      [   6   61   58  949    0    0]]

    On normalizing, the accuracy and precision dropped and the confusion matrix shows more data imbalance. 


3. **Question 3**

    We use the same models as we did in Task 3 to maintain consistency. We used one example of every activity class from the raw accelerometer data to train the model. We query the model on our collected data. We chose this specific data from the UCI-HAR dataset to match the data shape. The collected data was preprocessed by normalizing the data to better match the UCI-HAR dataset.<br>
    The `llama3.1-8b` model was able to predict only one test sample correctly, while the `llama3.1-70b` model predicted 3 out of 6 correctly. The metrics are as such<br>
    `llama3.1-8b :`<br>
    `Accuracy: 0.17, Precision: 0.08, Recall: 0.17`<br>
    `llama3.1-70b :`<br>
    `Accuracy: 0.50, Precision: 0.33, Recall: 0.50`<br>
    We repeated the experiment with 2 examples per activity class for training, and the performance of both the models dropped. The `llama3.1-8b` model predicted all test samples incorrectly, while the `llama3.1-70b` model predicted only one correctly.<br>
    `llama3.1-8b :`<br>
    `Accuracy: 0.00, Precision: 0.00, Recall: 0.00`<br>
    `llama3.1-70b :`<br>
    `Accuracy: 0.17, Precision: 0.04, Recall: 0.17`<br>

4. **Question 4**

    We use the same models as we did in Task 3 to maintain consistency. We used one example of every activity class from our collected data to train the model. We query the model on our collected data. The data collected from Manav was used to train the model, while the data collected from Arul was used to test it. This was done to ensure there is no data leaking.<br>
    The `llama3.1-8b` model had the same performace as with the model trained on the UCI-HAR dataset, but the `llama3.1-70b` model's performace significantly improved as such<br>
    `llama3.1-8b :`<br>
    `Accuracy: 0.50, Precision: 0.31, Recall: 0.50`<br>
    `llama3.1-70b :`<br>
    `Accuracy: 0.83, Precision: 0.75, Recall: 0.83`<br>
