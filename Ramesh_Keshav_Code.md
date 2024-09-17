# QUESTION 1 (EDA)


```python
import pandas as pd

permits_df = pd.read_csv('/Users/admin/Projects/WAF/DataChallenge/Building_Permits.csv', low_memory = False)
permits_df.shape
permits_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 198900 entries, 0 to 198899
    Data columns (total 43 columns):
     #   Column                                  Non-Null Count   Dtype  
    ---  ------                                  --------------   -----  
     0   Permit Number                           198900 non-null  object 
     1   Permit Type                             198900 non-null  int64  
     2   Permit Type Definition                  198900 non-null  object 
     3   Permit Creation Date                    198900 non-null  object 
     4   Block                                   198900 non-null  object 
     5   Lot                                     198900 non-null  object 
     6   Street Number                           198900 non-null  int64  
     7   Street Number Suffix                    2216 non-null    object 
     8   Street Name                             198900 non-null  object 
     9   Street Suffix                           196132 non-null  object 
     10  Unit                                    29479 non-null   float64
     11  Unit Suffix                             1961 non-null    object 
     12  Description                             198610 non-null  object 
     13  Current Status                          198900 non-null  object 
     14  Current Status Date                     198900 non-null  object 
     15  Filed Date                              198900 non-null  object 
     16  Issued Date                             183960 non-null  object 
     17  Completed Date                          97191 non-null   object 
     18  First Construction Document Date        183954 non-null  object 
     19  Structural Notification                 6922 non-null    object 
     20  Number of Existing Stories              156116 non-null  float64
     21  Number of Proposed Stories              156032 non-null  float64
     22  Voluntary Soft-Story Retrofit           35 non-null      object 
     23  Fire Only Permit                        18827 non-null   object 
     24  Permit Expiration Date                  147020 non-null  object 
     25  Estimated Cost                          160834 non-null  float64
     26  Revised Cost                            192834 non-null  float64
     27  Existing Use                            157786 non-null  object 
     28  Existing Units                          147362 non-null  float64
     29  Proposed Use                            156461 non-null  object 
     30  Proposed Units                          147989 non-null  float64
     31  Plansets                                161591 non-null  float64
     32  TIDF Compliance                         2 non-null       object 
     33  Existing Construction Type              155534 non-null  float64
     34  Existing Construction Type Description  155534 non-null  object 
     35  Proposed Construction Type              155738 non-null  float64
     36  Proposed Construction Type Description  155738 non-null  object 
     37  Site Permit                             5359 non-null    object 
     38  Supervisor District                     197183 non-null  float64
     39  Neighborhoods - Analysis Boundaries     197175 non-null  object 
     40  Zipcode                                 197184 non-null  float64
     41  Location                                197200 non-null  object 
     42  Record ID                               198900 non-null  float64
    dtypes: float64(13), int64(2), object(28)
    memory usage: 65.3+ MB



```python
# Check missing values in each column and display columns where >20% of values are missing
missing_values = permits_df.isnull().sum() / len(permits_df) * 100
missing_values[missing_values > 20].sort_values(ascending = False)
```




    TIDF Compliance                           99.998994
    Voluntary Soft-Story Retrofit             99.982403
    Unit Suffix                               99.014077
    Street Number Suffix                      98.885872
    Site Permit                               97.305681
    Structural Notification                   96.519859
    Fire Only Permit                          90.534439
    Unit                                      85.178984
    Completed Date                            51.135747
    Permit Expiration Date                    26.083459
    Existing Units                            25.911513
    Proposed Units                            25.596280
    Existing Construction Type                21.802916
    Existing Construction Type Description    21.802916
    Proposed Construction Type                21.700352
    Proposed Construction Type Description    21.700352
    Number of Proposed Stories                21.552539
    Number of Existing Stories                21.510307
    Proposed Use                              21.336853
    Existing Use                              20.670689
    dtype: float64




```python
# Drop columns where more than 80% of values are missing
drop_cols = missing_values[missing_values > 80].index
permits_cleaned_df = permits_df.drop(columns = drop_cols)
permits_cleaned_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 198900 entries, 0 to 198899
    Data columns (total 35 columns):
     #   Column                                  Non-Null Count   Dtype  
    ---  ------                                  --------------   -----  
     0   Permit Number                           198900 non-null  object 
     1   Permit Type                             198900 non-null  int64  
     2   Permit Type Definition                  198900 non-null  object 
     3   Permit Creation Date                    198900 non-null  object 
     4   Block                                   198900 non-null  object 
     5   Lot                                     198900 non-null  object 
     6   Street Number                           198900 non-null  int64  
     7   Street Name                             198900 non-null  object 
     8   Street Suffix                           196132 non-null  object 
     9   Description                             198610 non-null  object 
     10  Current Status                          198900 non-null  object 
     11  Current Status Date                     198900 non-null  object 
     12  Filed Date                              198900 non-null  object 
     13  Issued Date                             183960 non-null  object 
     14  Completed Date                          97191 non-null   object 
     15  First Construction Document Date        183954 non-null  object 
     16  Number of Existing Stories              156116 non-null  float64
     17  Number of Proposed Stories              156032 non-null  float64
     18  Permit Expiration Date                  147020 non-null  object 
     19  Estimated Cost                          160834 non-null  float64
     20  Revised Cost                            192834 non-null  float64
     21  Existing Use                            157786 non-null  object 
     22  Existing Units                          147362 non-null  float64
     23  Proposed Use                            156461 non-null  object 
     24  Proposed Units                          147989 non-null  float64
     25  Plansets                                161591 non-null  float64
     26  Existing Construction Type              155534 non-null  float64
     27  Existing Construction Type Description  155534 non-null  object 
     28  Proposed Construction Type              155738 non-null  float64
     29  Proposed Construction Type Description  155738 non-null  object 
     30  Supervisor District                     197183 non-null  float64
     31  Neighborhoods - Analysis Boundaries     197175 non-null  object 
     32  Zipcode                                 197184 non-null  float64
     33  Location                                197200 non-null  object 
     34  Record ID                               198900 non-null  float64
    dtypes: float64(12), int64(2), object(21)
    memory usage: 53.1+ MB



```python
# Converting relevant date columns to datetime format and checking for duplicates
date_cols = ['Permit Creation Date', 'Filed Date', 'Issued Date', 
    'Completed Date', 'First Construction Document Date', 'Permit Expiration Date']

for col in date_cols:
    permits_df[col] = pd.to_datetime(permits_df[col], errors = 'coerce')

duplicates_count = permits_df.duplicated().sum()
numeric_descriptive_stats = permits_df.describe()
{"duplicates_count": duplicates_count, "numeric_descriptive_stats": numeric_descriptive_stats}
```

    /var/folders/dd/0vbwv_857dvfzm221ynnp1900000gn/T/ipykernel_18124/927741701.py:6: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      permits_df[col] = pd.to_datetime(permits_df[col], errors = 'coerce')
    /var/folders/dd/0vbwv_857dvfzm221ynnp1900000gn/T/ipykernel_18124/927741701.py:6: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      permits_df[col] = pd.to_datetime(permits_df[col], errors = 'coerce')
    /var/folders/dd/0vbwv_857dvfzm221ynnp1900000gn/T/ipykernel_18124/927741701.py:6: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      permits_df[col] = pd.to_datetime(permits_df[col], errors = 'coerce')
    /var/folders/dd/0vbwv_857dvfzm221ynnp1900000gn/T/ipykernel_18124/927741701.py:6: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      permits_df[col] = pd.to_datetime(permits_df[col], errors = 'coerce')
    /var/folders/dd/0vbwv_857dvfzm221ynnp1900000gn/T/ipykernel_18124/927741701.py:6: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      permits_df[col] = pd.to_datetime(permits_df[col], errors = 'coerce')
    /var/folders/dd/0vbwv_857dvfzm221ynnp1900000gn/T/ipykernel_18124/927741701.py:6: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      permits_df[col] = pd.to_datetime(permits_df[col], errors = 'coerce')





    {'duplicates_count': 111,
     'numeric_descriptive_stats':          Permit Type           Permit Creation Date  Street Number  \
     count  198900.000000                         198900  198900.000000   
     mean        7.522323  2015-08-29 05:19:34.805429760    1121.728944   
     min         1.000000            2012-03-28 00:00:00       0.000000   
     25%         8.000000            2014-05-30 00:00:00     235.000000   
     50%         8.000000            2015-09-09 00:00:00     710.000000   
     75%         8.000000            2016-12-06 00:00:00    1700.000000   
     max         8.000000            2018-02-23 00:00:00    8400.000000   
     std         1.457451                            NaN    1135.768948   
     
                    Unit                     Filed Date  \
     count  29479.000000                         198900   
     mean      78.517182  2015-08-29 09:30:48.977375488   
     min        0.000000            2013-01-02 00:00:00   
     25%        0.000000            2014-05-30 00:00:00   
     50%        0.000000            2015-09-09 00:00:00   
     75%        1.000000            2016-12-06 00:00:00   
     max     6004.000000            2018-02-23 00:00:00   
     std      326.981324                            NaN   
     
                              Issued Date                 Completed Date  \
     count                         183960                          97191   
     mean   2015-09-02 14:00:18.786692864  2015-10-30 08:28:09.934252800   
     min              2013-01-02 00:00:00            2013-01-04 00:00:00   
     25%              2014-06-06 00:00:00            2014-08-29 00:00:00   
     50%              2015-09-16 00:00:00            2015-11-20 00:00:00   
     75%              2016-12-06 00:00:00            2017-01-09 00:00:00   
     max              2018-02-23 00:00:00            2018-02-23 00:00:00   
     std                              NaN                            NaN   
     
           First Construction Document Date  Number of Existing Stories  \
     count                           183954               156116.000000   
     mean     2015-09-04 13:35:18.386118400                    5.705773   
     min                2013-01-02 00:00:00                    0.000000   
     25%                2014-06-09 00:00:00                    2.000000   
     50%                2015-09-17 00:00:00                    3.000000   
     75%                2016-12-09 00:00:00                    4.000000   
     max                2018-02-23 00:00:00                   78.000000   
     std                                NaN                    8.613455   
     
            Number of Proposed Stories  ... Estimated Cost  Revised Cost  \
     count               156032.000000  ...   1.608340e+05  1.928340e+05   
     mean                     5.745043  ...   1.689554e+05  1.328562e+05   
     min                      0.000000  ...   1.000000e+00  0.000000e+00   
     25%                      2.000000  ...   3.300000e+03  1.000000e+00   
     50%                      3.000000  ...   1.100000e+04  7.000000e+03   
     75%                      4.000000  ...   3.500000e+04  2.870750e+04   
     max                     78.000000  ...   5.379586e+08  7.805000e+08   
     std                      8.613284  ...   3.630386e+06  3.584903e+06   
     
            Existing Units  Proposed Units       Plansets  \
     count   147362.000000   147989.000000  161591.000000   
     mean        15.666164       16.510950       1.274650   
     min          0.000000        0.000000       0.000000   
     25%          1.000000        1.000000       0.000000   
     50%          1.000000        2.000000       2.000000   
     75%          4.000000        4.000000       2.000000   
     max       1907.000000     1911.000000    9000.000000   
     std         74.476321       75.220444      22.407345   
     
            Existing Construction Type  Proposed Construction Type  \
     count               155534.000000               155738.000000   
     mean                     4.072878                    4.089529   
     min                      1.000000                    1.000000   
     25%                      3.000000                    3.000000   
     50%                      5.000000                    5.000000   
     75%                      5.000000                    5.000000   
     max                      5.000000                    5.000000   
     std                      1.585756                    1.578766   
     
            Supervisor District        Zipcode     Record ID  
     count        197183.000000  197184.000000  1.989000e+05  
     mean              5.538403   94115.500558  1.162048e+12  
     min               1.000000   94102.000000  1.293532e+10  
     25%               3.000000   94109.000000  1.308570e+12  
     50%               6.000000   94114.000000  1.371840e+12  
     75%               8.000000   94122.000000  1.435000e+12  
     max              11.000000   94158.000000  1.498340e+12  
     std               2.887041       9.270131  4.918216e+11  
     
     [8 rows x 21 columns]}




```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

permits_cleaned_df = permits_df.drop_duplicates()

# Plotting distribution of some numeric variables to check for outliers
plt.figure(figsize = (15, 10))

# Estimated Cost
plt.subplot(2, 2, 1)
sns.boxplot(data = permits_cleaned_df, x = 'Estimated Cost')
plt.title('Distribution of Estimated Cost')

# Number of Existing Stories
plt.subplot(2, 2, 2)
sns.boxplot(data = permits_cleaned_df, x = 'Number of Existing Stories')
plt.title('Distribution of Number of Existing Stories')

# Revised Cost
plt.subplot(2, 2, 3)
sns.boxplot(data = permits_cleaned_df, x = 'Revised Cost')
plt.title('Distribution of Revised Cost')

# Number of Proposed Stories
plt.subplot(2, 2, 4)
sns.boxplot(data = permits_cleaned_df, x = 'Number of Proposed Stories')
plt.title('Distribution of Number of Proposed Stories')

plt.tight_layout()
plt.show()
```


    
![png](output_5_0.png)
    


# QUESTION 2 (CLASSIFICATION)


```python
# Import necessary libraries for model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Feature Selection - choosing relevant columns for prediction
features = ['Permit Type', 'Estimated Cost', 'Revised Cost', 'Number of Existing Stories', 'Number of Proposed Stories', 
    'Existing Construction Type', 'Proposed Construction Type', 'Supervisor District', 'Neighborhoods - Analysis Boundaries']

# Select target column (Approved vs Withdrawn) and split dataset into features and target
permits_cleaned_df = permits_cleaned_df[permits_cleaned_df['Current Status'].isin(['approved', 'withdrawn'])]
X = permits_cleaned_df[features]
y = permits_cleaned_df['Current Status'].map({'approved': 1, 'withdrawn': 0})

# One-hot encoding categorical features
X_encoded = pd.get_dummies(X, drop_first = True)

# Train-test split and feature scaling
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest classifier model
clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

report = classification_report(y_test, y_pred, target_names = ['Withdrawn', 'Approved'])
print(report)
```

                  precision    recall  f1-score   support
    
       Withdrawn       0.98      0.95      0.97       537
        Approved       0.89      0.96      0.92       210
    
        accuracy                           0.95       747
       macro avg       0.94      0.96      0.94       747
    weighted avg       0.96      0.95      0.95       747
    



```python
from sklearn.impute import SimpleImputer

# Impute missing values for remaining NaNs
imputer = SimpleImputer(strategy = 'most_frequent')
X_encoded_imputed = imputer.fit_transform(X_encoded)

# Train-test split (re-splitting with imputed data) and re-train Random Forest model
X_train_imputed, X_test_imputed, y_train_imputed, y_test_imputed = train_test_split(X_encoded_imputed, y, test_size = 0.3, random_state = 42)

clf.fit(X_train_imputed, y_train_imputed)

# Feature importances
importances = clf.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
feature_importances_df = feature_importances_df.sort_values(by = 'Importance', ascending = False)
feature_importances_df.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Revised Cost</td>
      <td>0.554601</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Permit Type</td>
      <td>0.193672</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Estimated Cost</td>
      <td>0.081697</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Number of Proposed Stories</td>
      <td>0.029104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Number of Existing Stories</td>
      <td>0.026842</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Supervisor District</td>
      <td>0.023989</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Proposed Construction Type</td>
      <td>0.009681</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Existing Construction Type</td>
      <td>0.009063</td>
    </tr>
  </tbody>
</table>
</div>



# Question 3 (REGRESSION)


```python
from sklearn.impute import SimpleImputer

# Impute missing values for both numeric and categorical features
imputer = SimpleImputer(strategy='most_frequent')
X_encoded_reg_imputed = imputer.fit_transform(X_encoded_regression)

# Train-test split (re-splitting with imputed data) and re-train Linear Regression model
X_train_reg_imputed, X_test_reg_imputed, y_train_reg_imputed, y_test_reg_imputed = train_test_split(X_encoded_reg_imputed, y_regression, test_size = 0.3, random_state = 42)

lr_model.fit(X_train_reg_imputed, y_train_reg_imputed)

# Predictions and model evaluation
y_pred_reg_imputed = lr_model.predict(X_test_reg_imputed)

mse_imputed = mean_squared_error(y_test_reg_imputed, y_pred_reg_imputed)
r2_imputed = r2_score(y_test_reg_imputed, y_pred_reg_imputed)
mae_imputed = mean_absolute_error(y_test_reg_imputed, y_pred_reg_imputed)

mse_imputed, r2_imputed, mae_imputed
```




    (1003604183192.3921, 0.9303469908970614, 250394.65373094572)



# Visualizations


```python
# Plotting feature importance (Slide 2)
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df['Feature'][:8], feature_importances_df['Importance'][:8], color = 'orange')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top Features Influencing Permit Approval/Withdrawal')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.tight_layout()
plt.show()
```


    
![png](output_12_0.png)
    



```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Create confusion matrix (Slide 3)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Withdrawn', 'Approved'])
disp.plot(cmap = plt.cm.Blues)
plt.title('Confusion Matrix for Permit Approval Prediction')
plt.show()

report = classification_report(y_test, y_pred, target_names=['Withdrawn', 'Approved'])
print(report)
```


    
![png](output_13_0.png)
    


                  precision    recall  f1-score   support
    
       Withdrawn       0.98      0.95      0.97       537
        Approved       0.89      0.96      0.92       210
    
        accuracy                           0.95       747
       macro avg       0.94      0.96      0.94       747
    weighted avg       0.96      0.95      0.95       747
    



```python
# Scatterplot for actual vs. predicted costs for factors affecting construction costs (Slide 4)
plt.figure(figsize = (10, 6))
plt.scatter(y_test_reg_imputed, y_pred_reg_imputed, alpha=0.6, color = 'blue')
plt.plot([y_test_reg_imputed.min(), y_test_reg_imputed.max()], [y_test_reg_imputed.min(), y_test_reg_imputed.max()], 'k--', lw = 3, color = 'red')
plt.xlabel('Actual Revised Costs')
plt.ylabel('Predicted Revised Costs')
plt.title('Actual vs Predicted Construction Costs (Linear Regression)')
plt.tight_layout()
plt.show()
```

    /var/folders/dd/0vbwv_857dvfzm221ynnp1900000gn/T/ipykernel_18124/909003997.py:4: UserWarning: color is redundantly defined by the 'color' keyword argument and the fmt string "k--" (-> color='k'). The keyword argument will take precedence.
      plt.plot([y_test_reg_imputed.min(), y_test_reg_imputed.max()], [y_test_reg_imputed.min(), y_test_reg_imputed.max()], 'k--', lw = 3, color = 'red')



    
![png](output_14_1.png)
    



```python
# Summary table (Slide 5)
data = {"Key Metric": ["Accuracy (Approval)", "F1-Score (Approved)", "R-squared (Cost Prediction)", "MAE (Cost Prediction)"], "Value": ["95%", "0.92", "0.93", "$250,394"]}
df_summary = pd.DataFrame(data)

# Plot table
fig, ax = plt.subplots(figsize = (8, 2))
ax.axis('tight'), ax.axis('off')
table = ax.table(cellText = df_summary.values, colLabels = df_summary.columns, cellLoc = 'center', loc = 'center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)
plt.title('Summary of Key Metrics')
plt.show()
```


    
![png](output_15_0.png)
    

