# CMPT-353-Project

# Description
Given our initial analysis of the data, we will build a predictive model that prospective
employees who are entering the job market can use to take a look at the median wage of those
who have similar credentials to them and ensure that they have the upper hand when it comes to
contract negotiations and that they don’t sell themselves to short.

Hiring managers and headhunters could also make use of this model to guarantee that they offer
competitive incentives to the talent that they are looking to attract. Finally, the model will be
used to answer the question “Does your gender affect your wage?”

# Running the Analysis

The analysis section of the project is an jupyter notebook file. Therefore, in order to run the file a jupyter environment is necessary.

Once equipped witht the jupyter environment, all the cells of the notebook can be executed as long as the required python libraries provided below are all installed.

# Running the Models

There are three different models that were created to asses the accuracy of these different models and pick the method that is the most accurate for the final prediction model.  Running the commands in the order that they are listed in would allow for a better understanding of how the model was constructed and how it works.

1) To run the different models: 1) python3 Linear_Regression.py
                                
                                2) python3 Gradient_Boosting.py
                                
                                3) python3 RandomForest.py
    
    Each model will show the effect being a male or female had on it (coefficient of the feature), along with the predicted salary for man and a woman with all other factors staying the same.

2) The final prediction model amkes use of the RandomForest model, as that was found to be the most accurate. It takes an input csv file and gives you an output csv file
It can be run by the command: python3 Prediction.py input.csv output

Provided in the repo with the input.csv file, that can be replaced along with a sample/expected output file.

# Required Python libraries
Pandas: For data manipulation and analysis.

Scikit-learn (sklearn): For machine learning models and preprocessing tools.

NumPy: For numerical computations and array operations.

Scipy: For statistical tests

statsmodels: For generating the Q-Q plot

Matplotlib: For generating plots

sys (Built-in)

# Documentaions used/ References
1) https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

2) https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

3) https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

4) https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

5) https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

6) https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

7) https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

8) https://numpy.org/doc/

9) https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

10) https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

11) https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html 

12) https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html

13) https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html

14) https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html

15) https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
