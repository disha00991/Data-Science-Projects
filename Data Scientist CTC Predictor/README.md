![Python](https://img.shields.io/badge/Python-3.7.9-brightgreen) ![sklearn](https://img.shields.io/badge/sklearn-library-yellowgreen.svg) ![Pandas](https://img.shields.io/badge/pandas-library-green.svg) ![Seaborn](https://img.shields.io/badge/seaborn-library-orange.svg) ![tableau](https://img.shields.io/badge/Tableau-software-lightgrey)

## Project Overview
• Created a machine learning model that **estimates salary of data scientist from features like revenue, job title, number of competitors, etc.**
Checkout this app here: https://predict-ctc.herokuapp.com/

## How will this project help?
• This project can be **used by people in  data science and related fields to know their market rate and hence be prepared to negotiate their salary with their employers**

## Resources Used
* Packages: **pandas, numpy, sklearn, matplotlib, seaborn, pickle, Tableau, Tableau prep builder**<br/>
* Dataset by **Ken Jee**: https://github.com/PlayingNumbers/ds_salary_proj
* Visualization Techniques by **Josh**: https://www.kaggle.com/joshuaswords/data-visualization-clustering-mall-data
* Graph ideas by **Python Graph Gallery**: https://www.python-graph-gallery.com/
* Web app design idea by **Anuj Vyas**: https://github.com/anujvyas/Diabetes-Prediction-Deployment

## Exploratory Data Analysis (EDA) and Data Cleaning
• **Removed unwanted columns using Tableau prep builder**: 'Unnamed: 0'<br/>
<img src="readme-resources/remove_unwanted_cols.png" alt="drawing" width="600"/><br/>

• **Plotted bargraphs and countplots** for gathering insight into features<br/>
• **Removed unwanted alphabet/special characters from Salary feature**<br/>
• **Numerical Features** (Rating, Founded): **Replaced NaN or -1 values with mean or meadian based on their distribution**<br/>
<img src="readme-resources/replace_missing.png" alt="drawing" width="600"/><br/>

• **Categorical Features: Replaced NaN or -1 values with 'Other'/'Unknown' category**<br/>
• **Make the Salary column show one measure** i.e from (per hour, per annum, employer provided salary) to (per annum)

## Feature Engineering
• **Creating new features** from existing features e.g. **job_in_headquaters from (job_location, headquarters)**, etc.<br/>
• Creating separate columns for different skills i.e. **Python, excel, sql, tableau** to depict job focus<br/>
• **Reducing the number of categories** in columns like "job location", sector by **selecting top K categories, replacing rest by "others"** to reduce noise<br/>
![sample_counts_sector](readme-resources/sample_counts_sector.png)<br/>
![top_15_location](readme-resources/top_15_location.png)<br/>
• Feature Selection using **information gain (mutual_info_regression) and correlation matrix**<br/>
<img src="readme-resources/size_corr_matrix.png" alt="drawing" width="600"/><br/>
• Feature Scaling using **StandardScalar** such that final dataset is clean for evaluating on different models
<img src="readme-resources/final_df.png" alt="drawing" width="600"/>

## Model Building and Evaluation
Root Mean Squared Error (RMSE) is least for **Random Forest: ~17** out of the following tried algorithms (evaluated with cross validation on 10 folds)<br/>

* Multiple Linear Regression: ~27<br/>
* Lasso Regression: ~28<br/>
* **Random Forest: ~17**<br/>
* Gradient Boosting: ~24<br/>
* Voting (Random Forest + Gradient Boosting): ~19<br/>

## Model Hyper Parameter Tuning
Found best estimator from the following parameter grid (corresponding to the params in sklearn.ensemble.RandomForestRegressor):
* 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
* 'max_features': ['auto', 'sqrt']
* 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None]
* 'min_samples_split': [2, 5, 10]
* 'min_samples_leaf': [1, 2, 4]
* 'bootstrap': [True, False]

## Best Estimator:
![best_estimator](readme-resources/best_estimator.png)

## Saving the model and data prep objects
Used pickle to save the model and scaler transform objects for use during prediction from webapp.

## Deployment of Flask API to Heroku ![Heroku](https://img.shields.io/badge/Heroku-Deployment-blue.svg)
Created Flask app that:
* loads model and data prep objects from pickle files
* takes input features as input from user via WebApp deployed at https://predict-ctc.herokuapp.com/
* predicts salary!
<img src="readme-resources/user-input.png" alt="drawing" width="500"/>
<img src="readme-resources/predict.png" alt="drawing" width="500"/>
