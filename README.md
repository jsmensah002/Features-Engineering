Project Goal:
- Identify the most informative features for predicting the target variable (price). 

Method:
- Data Preparation: Collected and cleaned the dataset, handling missing values and encoding categorical variables.
- Feature Ranking: Feature importance was determined/ranked using Random Forest Regressor.
- Model Training: Trained linear regression, support vector regression, and Random Forest models incrementally, by adding features in threes to observe changes in model's performance.
- Evaluation: Monitored R² and signs of overfitting to assess model stability and predictive power.
- Feature Selection: Identified the top 12 features carrying most of the predictive signal; remaining features were classified as noise.

Full Details of the Results can be assessed from the excel file titled 'rankings compilation.xlsx'.

Key Insights:
- Generally, adding key features enhances the model’s ability to capture patterns in the data.
- The Random Forest model exhibited signs of overfitting as additional features were included, reducing its reliability. In contrast, the linear regression model remained stable, with performance (test 20% of data) plateauing beyond the top 12 features.
- The top 12 features contain the majority of the predictive information for the target, whereas the remaining features contribute minimally and can be considered as noise.
