# Assignment 1

## Task 1 : Exploratory Data Analysis (EDA)

## Task 2 : Decision Trees for Human Activity Recognition

1. **Use of Sklearn Library to Train Decision Trees**

    * **Results for decision tree model trained using the raw accelerometer data:**
    
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

        Here, from the confusion matrix, we can observe that the model predicts correctly for class 0. However, it isn't very accurate when it comes to the rest of the classes. To mitigate this inconsistency, we tried oversampling the minority class by applying SMOTE (Synthetic Minority Over-Sampling Technique). This implementation yields the following results:
    
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

    - Train Decision Trees with depths ranging from 2 to 8 for the raw accelerometer data, TSFEL features, and provided features.
    - Plot the accuracy of the model on test data versus the depth of the tree.

## Task 3 : Prompt Engineering for Large Language Models (LLMs)

## Task 4 : Data Collection in the Wild

## Decision Tree Implementation
