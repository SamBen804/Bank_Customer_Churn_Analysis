# Bank_Customer_Churn_Analysis

## Business Understanding

Stakeholder: Bank Owners and Managers that want to better understand if there is an identifiable pattern that we can find that will help to predict whether or not a customer will leave the bank.

## Data Understanding

The data used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn).

The dataset contains 18 columns housing 10,000 unique bank customer records. Generally speaking, the columns in this dataset contain various demographic measures, like age and country where the customer is living, along with recorded metrics that the bank is tracking like the Balance in each customer's account and how long the customer has held an account at the bank. A detailed description of each column was provided by the auther of the dataset, and I have copied it below:

#### Column Descriptions - from Kaggle:
- RowNumber—corresponds to the record (row) number and has no effect on the output.
- CustomerId—contains random values and has no effect on customer leaving the bank.
- Surname—the surname of a customer has no impact on their decision to leave the bank.
- CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
- Geography—a customer’s location can affect their decision to leave the bank.
- Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank.
- Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
- Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
- Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
- NumOfProducts—refers to the number of products that a customer has purchased through the bank.
- HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
- IsActiveMember—active customers are less likely to leave the bank.
- EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
- Exited—whether or not the customer left the bank.
- Complain—customer has complaint or not.
- Satisfaction Score—Score provided by the customer for their complaint resolution.
- Card Type—type of card hold by the customer.
- Points Earned—the points earned by the customer for using credit card.

## Data Preparation

### Exploratory Data Analysis Results
- The data does not have any null values to address.
- There are unneccessary columns that will need to be dropped:
    - RowNumber: This appears to be a duplicate index.
    - CustomerId: We are not analyzing specifc customer records, therefore we would not need to keep any information identifying individuals
    - Surname: similar information to the CustomerId column, we do not need individuals' information for this analysis.
- Certain 'object' type columns containing categorical variables need to be dealt with before being added to a model.
    - Geography: Spain, France, Germany
    - Gender: Male, Female
    - Card Type: DIAMOND, GOLD, SILVER, PLATINUM
- There are numeric columns that contain values that appear to be binary or their values may be ordinal.
    - Binary columns: 'HasCrCard', 'IsActiveMember', 'Exited', 'Complain' 
    - Ordinal column: 'Satisfaction Score'
        - Will need to treat this column more like a categorical variable than a numeric feature.
        
### Columns Dropped & Further Column Exploration
- I dropped redundant columns next - those that essentially repeated the index or had duplicate values.
- With all the remaining columns, I made observations by plotting the distirbution of values and/or checking the value counts for categorical and binary columns.

## Modeling

#### Target Identified: the 'Exited' column.
- A 0 represents a customer that DID NOT leave the bank.
    - It appears that about 80% of customers have remained at the bank.
- A 1 represents a customer that DID leave the bank AKA a customer that churned.
    - It appears that about 20% of customers have left the bank.
    
#### Exploring the Relationships between the target and other features:
- It appears that the 'Complain' column, which denotes whether or not a customer has filed a complaint (1 meaning Yes, a complaint was filed), almost perfectly correlates with the target. This is a KEY feature for this reason, and including this feature or not will likely have a major impact on how the models will learn to predict churn.

#### Splitting Categorical and Numeric Data
- Numeric data is readily able to be used in a model, whereas categorical data will require preporcessing. For the first model iterations, I will only use numeric data to get a baseline model.

### Model 1: Dummy - Numeric Data Only
The X values, or features, in this first model represent the independent variables that we want to measure as they effect our target value: y. Previously, the 'Exited' column was identified as the target, and in order to analysis the effect the features have on this target, we must isolate this column and assign it 'y'.

Next we will run a train_test_split, in order to separate our data into a training set that we can use to train our model, and a testing set, which will remain unseen until the end of this process, when a final model iteration is chosen.

Checking the value_counts and normalizing the results, below, gives us a look at the distribution of the positive (1s) and negative (0s) values contained in this column. Note that we will see a similar value throughout this first modeling process based on how we configure the dummy classifier model type.

Now we will instantiate 'model_1' as a DummyClassifier, setting the strategy to 'most_frequent' so we can get a very basic first model. Then we fit the model and check that the model is correctly predicting all 0's (the most frequent value in the target column). 

Checking the accuracy score, we find that this dummy model is able to predict that a customer IS NOT churning (0) at ~80% accuracy, which is similar, almost equal to the value we found running a value_counts on the 'Exited' column alone.

As expected, the cross validated score for this dummy model is ~80% too, meaning that even re-distirbuting the training data into multiple iterations within the original split of training data did not yield different results. Cross validating can often expose errors in splitting that can occur randomly when a train_test_split is performed. In this case, no such anomolys appear to be present.

To visualize the results of this model, we will plot a confusion matrix. We can see that the model accurately predicted 5982 customers as staying (0), while also incorrectly predicting that the other 1518 customers would stay, when they actually churned or left the Bank.

Another method to check how well our model is predicting values, is to plot a ROC curve - pictured below, this curve appears as a straight line becasue this model iteration is predicting ONLY the most frequent (0) values, the area under the curve is .5, the lowest it can be. 

### Model 2: Logisitc Regression - Numeric Features Only

We will use the same X and y values from above, as they have already been appropriately split into training and testing data that only includes numeric value columns.

The numeric data needs to be scaled so columns with vastly different numeric values - like a binary column containing just 1s and 0s, compared to the 'Balance' column containing dollar amounts in the thousands, can be combined into a single multi-feature model.

We will fit our Scaler to the training data ONLY, but still need to tranform our testing data so it aligns with what our model expects to see when we introduce the testing data to the model at the end of the process.

For model_2, we will be using logistic regression to model our features' effects on our target variable. This model should account for the individual feature's effects better than the dummy model above that made a generalized assumption that ALL values would negative in our target (0s). 

Checking the accuracy score of model_2 appears significantly high, with 1.0 being assigned to a 'perfect' model. This indicates that some mix of features (or one significantly correlated feature) are able to VERY accurately predict the value in the target column. Let's run a few additional tests to see if this model is just overfit or if it is actually performing at a near perfect level:

Visualizing this nearly perfect set of predictions in two different ways, to confirm the results found above. From viewing the confusion matrices, it appears that only 2 customers were predicted to stay when they actually left and only 7 customers were exepected to churn, when they actually stayed.

Cross validating our model's results corroborates our findings along with each of the subsequent scores below: 0.9987999999999999

A recall score of ~.9987 tells us that the model rarely (in this case only twice), incorrectly predicted that a customer would not churn, when in fact they did leave. This score is calcluated as a ratio of correct predictions of customers leaving (1516) divided by the total number of customers that ultimately left (1516 + 2).

A precision score of ~.9954 reflects that the model was slightly less able to correctly predict when a customer would churn, as effected by the 7 customers that the model predicted to leave, but did not. This score represents the number of customers correctly predicted to leave (1516) divided by the total number of customers predicted to leave (1516 + 7).

A summary of the training scores detailed above:

#### Model 2: Training Data Scores
-----------------------------
Accuracy:   0.9988
Recall:     0.9986824769433466
Precision:  0.9954038082731451
F1:         0.9970404472213088

### A near-perfect model? How?
- Recall from the EDA that the values in the 'Complain' column almost perfectly correlate with the values in the 'Exited' column. This linear relationship suggests that a 1 in the 'Complain' column almost always indicates that there will be a 1 in the 'Exited' column. 
- Said another way - it appears that nearly every time a customer complains, that customer then leaves or churns. 
- Naturally, having a perfect model sounds ideal, but from a business sense, what is this result suggesting?
    - One option, is to isolate this feature and further analyze the individual cases when a customer complained:
        - What caused the grievance?
        - Was the complaint addressed in a timely manner?
        - What improvements can be made to the customer experience to avoid future complaints and poor customer service received when a customer is frustrated?
    - Another option is to recognize the significance of this feature and complete the analysis above separate from the model, so we may be better able to understand if there are other features contributing to the likelihood that a customer will churn. Moving into our next model iterations, we will be dropping the 'Complain' column in order to best identify any additional features that may have larger impacts on the churn rate.

### Recommendation 1:
- When including the 'Complain' column data, we are able to almost perfectly predict whether or not a customer will churn.
- Essentially, if a customer complains, then they are VERY likely to leave the bank and should have their complaint addressed appropriately OR they will leave.

### Model 3: Logisitc Regression - Numeric Features Only + Dropping 'Complain'

We need to reassign, split and scale our data contained in the X features variable, with the 'Complain' column dropped to continue our analysis:

Checking the new accuracy score for this model, we immediately notice a difference. This accuracy score is much more simliar to our dummy model above. This score does not tell us how well our model is performing overall, however. In order to gain a better understanding, we need to check the effect on the recall and precision scores too and then cross-validate: 0.806

Visualizing this new model's predictive abilities shows that the model incorrectly predicted that 1299 customers would NOT leave, when in fact they did! 

Noting the relatively high amount of false negative predictions shown above, I want to cross validate the recall score, and see just how much this model iteration missed the mark. 
0.14755080771235018

#### Model 3: Training Data Scores
-----------------------------
Accuracy:   0.806
Recall:     0.1442687747035573
Precision:  0.584
F1:         0.23137876386687797

The accuracy score above is around 81% (a slight imprvement from model 1) because the model still appears to be functioning well to predict the custmers that will NOT leave, but that is not the goal. The company should not be as concerned with customers that are seemingly content.
- We're going to have to add more features and increase complexity to see if we can better-fit our model and raise this recall score!

### Model 4: Logistic Regression - Incorporation of Categorical Features, 'Complain' Column Remains Dropped
Now we need to create an X variable that includes all the columns, categorical AND numeric that still drops our target and the 'Complain' columns.

To appropriately preprocess the categorical variables in this dataset, I chose to OneHotEncode the columsn containing categorical information, convert it to a dataframe and then concatinate this data with scaled numeric data (less the 'Complain' column). Training and testing data underwent these same transformations in order to maintain consistency throughout the analysis.

For this model, we will use the default LogisticRegression parameters when we instantiate the model:

As we identified above, we want to focus on Recall, so I've cross-validated the recall score for this set of training data to ensure it aligns with the overall training data recall score of 0.2068568698975161

#### Model 4: Training Data Scores
-----------------------------
Accuracy:   0.8126666666666666
Recall:     0.20553359683794467
Precision:  0.6105675146771037
F1:         0.30754066042385414

Our model has improved on all of the scores when the categorical data was added. 
- the recall score being our focus, notably changed from .14 to .21. This is still not as high as we would like it to be, so we will continue tweaking the model parameters to raise this score

### Model 5: Logistic Regression - Tweaking the Penalty Parameter
For this model, I am changing the penalty parameter to 'none', which may or may not be effective with the smaller number of features we are analyzing, as an attempt to increase our recall score by minizing the number of False Negatives. 

#### Model 5: Training Data Scores
-----------------------------
Accuracy:   0.8126666666666666
Recall:     0.20619235836627142
Precision:  0.6101364522417154
F1:         0.3082225504677498

Comparing Model 5 to Model 4, there is very little (almost no) difference in the Recall score (or really any of the scores). Will have to try different methods to have a greater effect on Recall.

### Model 6: Logistic Regression - Reducing Regularization
For the 6th model, I will attempt to reduce regularization by increasing the C value parameter:

#### Model 6: Training Data Scores
-----------------------------
Accuracy:   0.8126666666666666
Recall:     0.20619235836627142
Precision:  0.6101364522417154
F1:         0.3082225504677498

Similar to Model 5 and Model 6, this was not an effective tunign strategy to increase the recall score.

### Model 7: Logistic Regression - Addressing Class Imbalance
Of note, the distribution of the positive and negative class is around 20 and 80 percent respectively. To address this class imbalance, I will change the class_weight parameter to balanced when instantiating model_7.

#### Model 7: Training Data Scores
-----------------------------
Accuracy:   0.7052
Recall:     0.6956521739130435
Precision:  0.3764705882352941
F1:         0.48854961832061067

The accuracy score of the training data run through thhis model is MUCH better and is an increase of around .49 from our initial model recall of .20.

In order to see the impact of the individual features acting within the model, I will now analyze the coefficients associated with each feature: 

### Model 7 = Final Model

Now we can run our testing data through this model to see how the model performs on unseen data:

#### Final Model: Testing Data Scores
-----------------------------
Accuracy:   0.7184
Recall:     0.676923076923077
Precision:  0.3963963963963964
F1:         0.5

## Evaluation

This final model recall score increased significantly from our initial model recall score of around .20. This increase is due to addressing the imbalance of the positive and negative classes.

Next I will visualize the impacts of the key features identified in the analysis by graphing the absolute values of each features' coefficient. Of course, there are more features, but I chose 4 of each - 1 set is contributing to a customer leaving (or the probability that they will churn) and the second set displayed shows the 4 features that are contributing to the probability that a customer will stay with the bank.

![Churn_Features](images/Churn_Features.png)

- Age has the greatest impact after complaint. It appears that the higher the age, the higher the probability that customer will churn

- Account Balance - higher the account balance, more likely to churn

![Retention_Features](images/Retention_Features.png)

We can reverse engineer some inferences based on these feature impacts, which initially had negative coefficients, but I have displayed their absolute value for ease of understanding (as thier magnitude is not changed by switching signs from negative to positive in this case!).
- Male customers are less likely to churn compared to Females
    - Does our marketing appeal more towards males or is it some other factor?
- Activity - the more active a customer, the less likely they are to churn
    - How can we increase customer engagement?
    
## Recommendations

1. Redesign customer complaint process
    - Build a more robust customer support network
2. Using this model on all customer records moving forward to predict if a customer may churn
    - Run all future data through this model to identify any key factors for a client to determine the probability that they will churn.
3. Develop programs that engage older customers and female customers, encouraging them to remain active members

## Next Steps

1. Conduct complaint analysis 
2. Address customer frustrations identified in complaint analysis
3. Analyze the cost of different customer service strategies
    - Email campaigns vs. Customer calls