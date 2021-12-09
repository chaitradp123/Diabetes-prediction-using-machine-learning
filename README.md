## Project Title:  Diabetes prediction using Machine Learning Models

Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar.
According to 2017 statistics, 425  million  people  suffered  from  diabetes.
The World Health Organization predicts that by 2030, diabetes will become the seventh leading cause of death in the world. 
There are few different types of diabetes:<br>
    - Type 1 diabetes is an autoimmune disease. The immune system attacks and destroys cells in the pancreas, where insulin
	  is made. It’s unclear what causes this attack. About 10 percent of people with diabetes have this type.<br>
	- Type 2 diabetes occurs when your body becomes resistant to insulin, and sugar builds up in your blood.<br>
	- Prediabetes occurs when your blood sugar is higher than normal, but it’s not high enough for a diagnosis of type 2 diabetes.<br>
	- Gestational diabetes is high blood sugar during pregnancy. Insulin-blocking hormones produced by the placenta cause this type of 	diabetes.


## YouTube presentation
The short video presentation for this project is at https://youtu.be/HX7jUxbFJxI


## Powerpoint presentation
Slide presentation is uploaded in the repository

## Project Description:
Predicting if patients have diabetes or not using machine learning model


## Data set description
Data source : https://www.kaggle.com/ilkeryildiz/diabetes-prediction-using-machine-learning/data
 <br>
Data set 	: Pima indian daibetes dataset
* Features planned to use initially :   
	- Pregnancies: Number of times pregnant
	- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
	- BloodPressure: Diastolic blood pressure (mm Hg)
	- SkinThickness: Triceps skin fold thickness (mm)
	- Insulin: 2-Hour serum insulin (mu U/ml)
	- BMI: Body mass index (weight in kg/(height in m)^2)
	- DiabetesPedigreeFunction: Diabetes pedigree function
	- Age: Age (years)
	- Outcome: Class variable (0 or 1) 
## Pre-reqs
a) tool for python programming - we've used Jupyter notebook (recommended) <br>
b) packages:<br>
	- pandas			for importing dataset <br>
	- seaborn			data exploration <br>
	- sklearn			model building and prediciton <br>
	- matplotlib		visualizations <br>
	
## Running the code
1) make sure to have the '.csv' data file in the right folder as in code
2) if all packages are installed already, run the code as is, the order has been set.
3) Use the model to predict new data, if needed
4) model equation has been given for ease of use - can be performed manually as well.

## Section-wise code:
1) Data preparation
	- importng data set
	- removing redundant data
	- handling null / missing values
	- categorical data to numeric data
2) Data exploration
	- distribution of all the features
	- pairplot,scatterplot,pie chart
3) Feature selection
	- correlation table
	- sorting correlation values
4) Model building 
	-Logistic regression
	 The model gave an accuracy of 0.78. The confusion matrix 
	 showed that 87 patients were diagnosed as not having diabetes and 34 
	 patients are correctly predicted as having diabetes.<br>
	-K-Nearest neighbour
	 The accuracy of KNN was 0.71.To improve the accuracy of the model,
	 hyperparameters using Grid search was turned.<br>
	-Support vector machine
	 For this classifier,0.75 accuracy was achieved. Evaluation was done using 
	 confusion matrix and it predicted that 11 
	 healthy patients have diabetes.<br>
	-Decision tree 
	 The accuracy of the Decision tree gave an accuracy of 0.68.
	 Altering hyper parameters using Grid Search in Scikit-Learn. 
     Optimized max _depth = 7
     min_samples_leaf = 8.<br>
	-Random forest
	 Random forest model gave an accuracy of 0.79.
	 Altering hyper parameters using Grid Search in Scikit-Learn caused the model 
	 to have a slight improvement in accuracy with a FN 14 and FP 17.<br>
	-XGBoost
	 XGBoost(Extreme Gradient Boosting) which is an ensemble method gave an
	 accuracy of 0.80 which is better than all the models we have built thus far.<br> 
	-Light GBM
	 Light GBM(LGBM) is a fast, distributed, high performing gradient boosting 
	 framework which is based on decision tree, used for classification and other
	 machine learning tasks.
	 The accuracy of the model was 0.78.<br>
5) Model evaluation 
		- hyperparameter tuning<br>
		- plotting confusion matrix <br>
		- plotting ROC curve <br>
6) Prediction
	- using the best model from above analysis (XGBoost)
	- predicting values for test data set
7) Future work
    - use of larger dataset and features to build a better model.
## References
	https://www.sciencedirect.com/science/article/pii/S1877050920300557<br>
	https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset/blob/master/Pima%20Indians%20Diabetes%20Dataset.ipynb<br>
	https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/<br>


