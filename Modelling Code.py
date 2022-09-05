
#importing the required libraries and setting the path of the working directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import random
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
#get_ipython().system('pwd')

#Reading csv
completed_session = pd.read_csv(path)
abandoned_session = pd.read_csv(path)

#Dropping columns
completed_session = completed_session.drop(columns = ['domain','attributes/consumerName','organisationUuid','attributes/formDefinitionId','attributes/onlineEntryMethodCode','timeStamp', 'attributes/journeyType','attributes/doubleClickId','formSlug','formUuid'])
abandoned_session = abandoned_session.drop(columns = {'domain','attributes/consumerName','organisationUuid','attributes/formDefinitionId','attributes/onlineEntryMethodCode','timeStamp', 'attributes/journeyType','attributes/doubleClickId','formSlug','formUuid'})

# Correcting data quality issues
completed_session.loc[completed_session['field/tagName'] == 'SELECT', 'field/type'] = 'dropdown'
abandoned_session.loc[abandoned_session['field/tagName'] == 'SELECT', 'field/type'] = 'dropdown'
completed_session.loc[completed_session['field/id'] == 'apply-screen:keeping-in-touch:future-soft-search-permission', 'field/name'] = 'future-soft-search-permission'
abandoned_session.loc[abandoned_session['field/id'] == 'apply-screen:keeping-in-touch:future-soft-search-permission', 'field/name'] = 'future-soft-search-permission'
completed_session.loc[completed_session['field/id'] == 'apply-screen:keeping-in-touch:marketing-permission', 'field/name'] = 'marketing-permission'
abandoned_session.loc[abandoned_session['field/id'] == 'apply-screen:keeping-in-touch:marketing-permission', 'field/name'] = 'marketing-permission'

# Extracting field names
def get_fieldname(field):
    return field.split(':')[-1]

completed_session['field_name'] = completed_session['field/name'].dropna().apply(get_fieldname)
abandoned_session['field_name'] = abandoned_session['field/name'].dropna().apply(get_fieldname)

# Grouping the questions according to their steps
def get_formstep(field):
    group = field.split(':')[0]
    field_name = field.split(':')[-1]
    if field_name in ['bt-amount', 'interested-in-bt', 'title','date-of-birth-day','date-of-birth-month','date-of-birth-year', 'email-address', 'first-name', 'last-name']:
        return 1
    if field_name in ['date-of-birth-day', 'date-of-birth-month', 'date-of-birth-year', 'email-address', 'home-number', 'mobile-number']:
        return 2
    if field_name in ['employment-status', 'occupation', 'yearly-income','other-household-income']:
        return 3
    if field_name in ['number-of-dependants', 'cash-advance', 'rent-mortgage-payments', 'residential-status']:
        return 4
    if field_name in ['address-line-2', 'county', 'flat-number', 'house-name', 'house-number', 'list', 'postcode', 'street-name', 'town-city', 'years-at-address', 'address-line-2', 'county', 'flat-number', 'house-name', 'house-number', 'list', 'postcode', 'street-name', 'town-city', 'years-at-address','current-address-lookup-house-number','current-address-lookup-postcode','previous-address-lookup-house-number','previous-address-lookup-postcode']:
        return 5
    if field_name in ['marketing-permission','future-soft-search-permission']:
        return 6
completed_session['field_step'] = completed_session['field/name'].dropna().apply(get_formstep)
abandoned_session['field_step'] = abandoned_session['field/name'].dropna().apply(get_formstep)

question_sequence = {'title':1, 'first-name' : 2,'last-name' : 3,'date-of-birth-day' : 4, 'date-of-birth-month' : 5, 'date-of-birth-year' : 6,'email-address' : 7,
                     'mobile-number' : 8,  'home-number' : 9, 'employment-status' : 10, 'occupation' : 11, 'yearly-income' : 12, 'other-household-income' : 13, 'number-of-dependants' : 14,'cash-advance' : 15,
                     'residential-status' : 16,  'rent-mortgage-payments' : 17,'current-address-lookup-postcode' : 18, 'current-address-lookup-house-number' : 19,'house-number' : 20, 'house-name' : 21, 
                     'flat-number' : 22,'street-name' : 23, 'address-line-2' : 24,'town-city' : 25,'county' : 26,'postcode' : 27, 'years-at-address' : 28,'previous-address-lookup-postcode' : 29,
                     'previous-address-lookup-house-number' : 30, 'marketing-permission':31, 'future-soft-search-permission':32}

def get_fieldNumber(field):
    if field in question_sequence:
        return question_sequence[field]
completed_session['field_number'] = completed_session['field_name'].dropna().apply(get_fieldNumber)
abandoned_session['field_number'] = abandoned_session['field_name'].dropna().apply(get_fieldNumber)

# Removing inconsistent fields to calculate field returns
a1=completed_session.loc[completed_session['field/name'] == 'landing-screen:balance-transfer:bt-amount']
b1=completed_session.loc[completed_session['field/name'] == 'landing-screen:balance-transfer:interested-in-bt']
c1=completed_session.loc[completed_session['field/name'] == 'landing-screen:email-address:email-address']
d1=completed_session.loc[completed_session['field/name'] == 'about-you:date-of-birth:date-of-birth-day']
e1=completed_session.loc[completed_session['field/name'] == 'about-you:date-of-birth:date-of-birth-month']
f1=completed_session.loc[completed_session['field/name'] == 'about-you:date-of-birth:date-of-birth-year']
a2=abandoned_session.loc[abandoned_session['field/name'] == 'landing-screen:balance-transfer:bt-amount']
b2=abandoned_session.loc[abandoned_session['field/name'] == 'landing-screen:balance-transfer:interested-in-bt']
c2=abandoned_session.loc[abandoned_session['field/name'] == 'landing-screen:email-address:email-address']
d2=abandoned_session.loc[abandoned_session['field/name'] == 'about-you:date-of-birth:date-of-birth-day']
e2=abandoned_session.loc[abandoned_session['field/name'] == 'about-you:date-of-birth:date-of-birth-month']
f2=abandoned_session.loc[abandoned_session['field/name'] == 'about-you:date-of-birth:date-of-birth-year']
merge_questions_completed = pd.concat([a1,b1,c1,d1,e1,f1], axis=0)
merge_questions_abandoned = pd.concat([a2,b2,c2,d2,e2,f2], axis=0)
completed_visitorid_drop=merge_questions_completed['visitorId'].unique()
abandoned_visitorid_drop=merge_questions_abandoned['visitorId'].unique()
completed_session=completed_session[completed_session.visitorId.isin(completed_visitorid_drop)==False]
abandoned_session=abandoned_session[abandoned_session.visitorId.isin(abandoned_visitorid_drop)==False]

# Unique users in both sessions
unique_users_completed = pd.unique(completed_session['visitorId'])
count_users_completed = unique_users_completed.size
unique_users_abandoned = pd.unique(abandoned_session['visitorId'])
count_users_abandoned = unique_users_abandoned.size
print("Unique users in completed sessions are:",count_users_completed)
print("Unique users in abandoned sessions are:",count_users_abandoned)

# Randomising and setting seed
random.seed(42)
random.shuffle(unique_users_completed)
random.shuffle(unique_users_abandoned)

# Selecting forms of desired number until chosen step
desired_step_rows = abandoned_session.loc[abandoned_session['field_step'] == 1] #change number till selected step (1 to 6) to perform its modelling 
forms_abandoned_at_last_step = pd.unique(desired_step_rows['visitorId'])
if len(forms_abandoned_at_last_step) > 5000: #change number to change sample size
    forms_abandoned_at_last_step = forms_abandoned_at_last_step[:5000] #change number to change sample size

## Selecting sample size of 5000
selected_size = 5000 #change number to change sample size
unique_users_completed = unique_users_completed[:selected_size]
count_users_completed = selected_size
unique_users_abandoned = forms_abandoned_at_last_step[:selected_size]
count_users_abandoned = selected_size

# Unique field names in both sessiosn
unique_field_name_completed = pd.unique(completed_session['field_name'])
unique_field_name_completed = pd.Series(unique_field_name_completed).dropna()
count_unique_field_name_completed = len(unique_field_name_completed)
unique_field_name_abandoned = pd.unique(abandoned_session['field_name'])
unique_field_name_abandoned = pd.Series(unique_field_name_abandoned).dropna()
count_unique_field_name_abandoned = len(unique_field_name_abandoned)
print("Total number of question in completed sessions:",count_unique_field_name_completed )
print("Total number of question in abandoned sessions:", count_unique_field_name_abandoned)

# Calculating time spent from arival time 
# Completed
completed_session['time_spent_unix'] = np.nan
for visitor in unique_users_completed:
  prev=-1
  form_completed = completed_session.loc[completed_session['visitorId'] == visitor]
  form_completed = form_completed.sort_values('receivedAt')
  for index, row in form_completed.iterrows():
    if prev == -1:
      time_spent_unix = 0
    else:
      time_spent_unix = (row['receivedAt']-form_completed.loc[prev,('receivedAt')])/1000
    prev = index
    completed_session.loc[index,('time_spent_unix')] = time_spent_unix

# Abandoned
abandoned_session['time_spent_unix'] = np.nan
for visitor in unique_users_abandoned:
  prev=-1
  form_abandoned = abandoned_session.loc[abandoned_session['visitorId'] == visitor]
  form_abandoned = form_abandoned.sort_values('receivedAt')
  for index, row in form_abandoned.iterrows():
    if prev == -1:
      time_spent_unix = 0
    else:
      time_spent_unix = (row['receivedAt']-form_abandoned.loc[prev,('receivedAt')])/1000
    prev = index
    abandoned_session.loc[index,('time_spent_unix')] = time_spent_unix

# Checking field returns 
# Completed
completed_session['fieldreturns'] = np.nan
for visitor in unique_users_completed:
  prev=-1
  form_completed = completed_session.loc[completed_session['visitorId'] == visitor]
  form_completed = form_completed.sort_values('receivedAt')
  for index, row in form_completed.iterrows():
    if prev == -1:
      fieldreturns = 0
    else:
      fieldreturns = (row['field_number']-form_completed.loc[prev,('field_number')])
    prev = index
    completed_session.loc[index,('fieldreturns')] = fieldreturns

# Abandoned
abandoned_session['returns'] = np.nan
for visitor in unique_users_abandoned:
  prev=-1
  form_abandoned = abandoned_session.loc[abandoned_session['visitorId'] == visitor]
  form_abandoned = form_abandoned.sort_values('receivedAt')
  for index, row in form_abandoned.iterrows():
    if prev == -1:
      fieldreturns = 0
    else:
      fieldreturns = (row['field_number']-form_abandoned.loc[prev,('field_number')])
    prev = index
    abandoned_session.loc[index,('fieldreturns')] = fieldreturns

# Calculating Total Field Returns
def get_field_returns(rows):
    questions_returned_at = list(rows.loc[(rows['fieldreturns'] < 0)]['field_name'])
    unique_questions_returned_at = pd.unique(questions_returned_at)
    freq_question_returned_at = dict()
    for question in unique_questions_returned_at:
        freq_question_returned_at[question] = questions_returned_at.count(question)
    if freq_question_returned_at.items():
        return question_sequence[max(freq_question_returned_at.items(), key=operator.itemgetter(1))[0]]
    return 0

# Calculating Total Time Taken for a visitor session
def calculate_time_spent_unix(rows):
    time_spent_unix = rows['time_spent_unix'].sum()
    return time_spent_unix

# Calculating Text Field Counts for a visitor session
def count_text(rows):
    text = rows.loc[(rows['field/type'] == 'text')].shape[0]
    return text

# Calculating Field Returns for a visitor session
def get_looping_count(rows):
    count_looped = rows.loc[(rows['fieldreturns'] < 0)].size
    return count_looped

# Calculating Total Backspaces for a visitor session
def calculate_backspaces(rows):
    backspaces = rows.loc[(rows['key'] == 'Backspace')].shape[0]
    return backspaces

# Calculating Total Clicks for a visitor session
def calculate_clicks(rows):
    clicks =  rows.loc[rows['type']=='click'].shape[0]
    return clicks

completed_session = completed_session.loc[completed_session['visitorId'].isin(unique_users_completed)]
abandoned_session = abandoned_session.loc[abandoned_session['visitorId'].isin(unique_users_abandoned)]

# Getting stepwise information
def get_stats(data,step):
    data = data.loc[(data["field_step"] <= step)]
    stats = pd.DataFrame()
    unique_users = pd.unique(data["visitorId"])
    for user in unique_users:
        user_stats = dict()
        rows = data.loc[data["visitorId"] == user]
        step1_rows = rows.loc[rows['field_step'] == 1]
        user_stats["VisitorID"] = user
        user_stats["Time_Taken_till_Step"] = calculate_time_spent_unix(rows)
        #user_stats["Time_Taken_Step1"] = calculate_time_spent_unix(step1_rows)       #remove "#" when modelling for other steps
        user_stats["Backspaces_till_Step"] = calculate_backspaces(rows)
        user_stats["Clicks_till_Step"] = calculate_clicks(rows)
        #user_stats["Clicks_Step1"] = calculate_clicks(step1_rows)               #remove "#" when modelling for other steps
        user_stats["Text_Field_Count_till_Step"] = count_text(rows)
        user_stats["Field_Returns_till_Step"] = get_field_returns(rows)
        stats = pd.concat([stats,pd.DataFrame([user_stats])], axis=0, ignore_index=True)
    return stats

step = 1 #change number till selected step (1 to 6) to perform its modelling
stats_completed_users = get_stats(completed_session,step)
stats_abandoned_users = get_stats(abandoned_session,step)

ax = stats_completed_users.plot(kind='box', title='Characteristic features of completed features')
plt.xticks(rotation = 90)
plt.show()

ax = stats_abandoned_users.plot(kind='box', title='Characteristic features of abandoned features')
plt.xticks(rotation = 90)
plt.show()

# Removing Outliers (1.5*InterQuartileRange)
def remove_outliers(df, col):
    Q1 = np.percentile(df[col], 25, interpolation = 'midpoint')
    Q3 = np.percentile(df[col], 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    print(Q1,Q3)   
    upper = np.where(df[col] >= (Q3+1.5*IQR))
    lower = np.where(df[col] <= (Q1-1.5*IQR))
    df = df.drop(index = upper[0])
    df = df.drop(index = lower[0])
    return df.reset_index(drop=True)

for col in stats_completed_users.columns[1:2]: #make [1:3] for steps 2-6 as an additional time variable considered
    stats_completed_users = remove_outliers(stats_completed_users,col)

for col in stats_abandoned_users.columns[1:2]: #make [1:3] for steps 2-6 as an additional time variable considered
    stats_abandoned_users = remove_outliers(stats_abandoned_users,col)

ax = stats_completed_users.plot(kind='box', title='Characteristic features of completed features (Outliers Removed)')
plt.xticks(rotation = 90)
plt.show()

ax = stats_abandoned_users.plot(kind='box', title='Characteristic features of abandoned features (Outliers Removed)')
plt.xticks(rotation = 90)
plt.show()

#Setting binary labels
Online_Form_Abandonment = {0 : 'Completed',  1 : 'Abandond'}
stats_completed_users['Online_Form_Abandonment'] = 0
stats_abandoned_users['Online_Form_Abandonment'] = 1

#Merging completed and abandoned datasets and preparing for modelling 
merged_data = pd.concat([stats_completed_users, stats_abandoned_users], axis=0)
merged_data

merged_data.describe().T

merged_data.corr()

X = np.array(merged_data[['Time_Taken_till_Step','Backspaces_till_Step','Clicks_till_Step','Text_Field_Count_till_Step', 'Field_Returns_till_Step']]) #add 'Time_Taken_Step1' after 'Time_Taken_till_Step' and 'Clicks_Step1' after 'Clicks_till_Step' for other steps
y = np.ravel(np.array(merged_data[['Online_Form_Abandonment']]))
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Hyperparameter optimization
modelLR = LogisticRegression()
modelDT = tree.DecisionTreeClassifier()
modelRF = RandomForestClassifier()

paramsLR = {
    'solver' : ['sag','saga','lbfgs','liblinear']
         }
paramsDT = {
    'min_samples_split' : [2, 3, 4]
         }
paramsRF = {
    'n_estimators' : [50, 100, 150, 200, 250],
    'min_samples_split' : [2, 3, 4],
    'max_depth': [2,4,6,8,10]
         }

grid_modelLR = GridSearchCV(estimator = modelLR, 
                          param_grid = paramsLR, 
                         scoring = 'accuracy',
                         cv = 5, verbose = 2)
grid_modelDT = GridSearchCV(estimator = modelDT, 
                          param_grid = paramsDT, 
                         scoring = 'accuracy',
                         cv = 5, verbose = 2)
grid_modelRF = GridSearchCV(estimator = modelRF, 
                          param_grid = paramsRF, 
                         scoring = 'accuracy',
                         cv = 5, verbose = 2)

grid_modelLR.fit(X_train, y_train)

grid_modelDT.fit(X_train, y_train)

grid_modelRF.fit(X_train, y_train)

###Adjust model hyperparameters according to following output###

grid_modelLR.best_estimator_

grid_modelDT.best_estimator_

grid_modelRF.best_estimator_

## Logistic regression ##
clf = LogisticRegression(random_state=42, solver = 'sag').fit(X_train, y_train) #adjusting hyperparameters according to best_estimator_ results from above
y_pred_logistic = clf.predict(X_test)
y_pred_prob_logistic = clf.predict_proba(X_test)[::,1]

print(f" Score = {round(clf.score(X_test,y_test),3)}")
print(f" Predicted Completed Sessions = {len(y_pred_logistic[y_pred_logistic == 0])}")
print(f" Predicted Abandoned Sessions = {len(y_pred_logistic[y_pred_logistic == 1])}")
print(f" Accuracy = {sklearn.metrics.accuracy_score(y_test,y_pred_logistic)}")
print(f" Precision = {sklearn.metrics.precision_score(y_test,y_pred_logistic)}")
print(f" Recall = {sklearn.metrics.recall_score(y_test,y_pred_logistic)}")
print(f" F1 Score = {sklearn.metrics.f1_score(y_test,y_pred_logistic)}")

#Coefficients of Logistic Regression / Feature Importance
# get importance
importance = clf.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
model_importances = pd.Series(importance, index=merged_data.drop(["VisitorID", "Online_Form_Abandonment"], axis = 1).columns).to_frame().reset_index()
model_importances.columns = ['feature', 'importance']
sns.barplot(x = "feature", y = "importance", data = model_importances).set(title='Coefficients of Logistic Regression')
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

#Confusion Matrix
plot_confusion_matrix(clf,X_test, y_test)

## Decision tree ##
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=42,min_samples_split = 4) #adjusting hyperparameters according to best_estimator_ results from above
clf = clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
y_pred_prob_tree = clf.predict_proba(X_test)[::,1]

print(f" Score = {round(clf.score(X_test,y_test),3)}")
print(f" Predicted Completed Sessions = {len(y_pred_decision_tree[y_pred_decision_tree == 0])}")
print(f" Predicted Abandoned Sessions = {len(y_pred_decision_tree[y_pred_decision_tree == 1])}")
print(f" Accuracy = {sklearn.metrics.accuracy_score(y_test,y_pred_decision_tree)}")
print(f" Precision = {sklearn.metrics.precision_score(y_test,y_pred_decision_tree)}")
print(f" Recall = {sklearn.metrics.recall_score(y_test,y_pred_decision_tree)}")
print(f" F1 Score = {sklearn.metrics.f1_score(y_test,y_pred_decision_tree)}")

#Feature Importance
# get importance
importance = clf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
model_importances = pd.Series(importance, index=merged_data.drop(["VisitorID", "Online_Form_Abandonment"], axis = 1).columns).to_frame().reset_index()
model_importances.columns = ['feature', 'importance']

sns.barplot(x = "feature", y = "importance", data = model_importances).set(title='Decision Tree Feature Importance')
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

#Confusion Matrix
plot_confusion_matrix(clf,X_test, y_test)

## Random forest ##
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=8, random_state=42, n_estimators = 200, min_samples_split=3) #adjusting hyperparameters according to best_estimator_ results from above
clf = clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
y_pred_prob_random_forest = clf.predict_proba(X_test)[::,1]

print(f" Score = {round(clf.score(X_test,y_test),3)}")
print(f" Predicted Completed Sessions = {len(y_pred_random_forest[y_pred_random_forest == 0])}")
print(f" Predicted Abandoned Sessions = {len(y_pred_random_forest[y_pred_random_forest == 1])}")
print(f" Accuracy = {sklearn.metrics.accuracy_score(y_test,y_pred_random_forest)}")
print(f" Precision = {sklearn.metrics.precision_score(y_test,y_pred_random_forest)}")
print(f" Recall = {sklearn.metrics.recall_score(y_test,y_pred_random_forest)}")
print(f" F1 Score = {sklearn.metrics.f1_score(y_test,y_pred_random_forest)}")

#Feature Importance
# get importance
importance = clf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
model_importances = pd.Series(importance, index=merged_data.drop(["VisitorID", "Online_Form_Abandonment"], axis = 1).columns).to_frame().reset_index()
model_importances.columns = ['feature', 'importance']
sns.barplot(x = "feature", y = "importance", data = model_importances).set(title='Random Forest Feature Importance')
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

#Confusion Matrix
plot_confusion_matrix(clf,X_test, y_test)

###ROC and AUC### 
def compute_ROC(y_score, y_test):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
def plot_ROC(fpr,tpr,roc_auc,title):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="orangered",
        lw=lw,
        label="ROC curve (AUC = %0.3f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="blue", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

fpr, tpr, roc_auc = compute_ROC(y_pred_prob_logistic, y_test)
plot_ROC(fpr,tpr,roc_auc,"ROC Curve - Logistic Regression")

fpr, tpr, roc_auc = compute_ROC(y_pred_prob_tree, y_test)
plot_ROC(fpr,tpr,roc_auc,"ROC Curve -  Decision Tree")

fpr, tpr, roc_auc = compute_ROC(y_pred_prob_random_forest, y_test)
plot_ROC(fpr,tpr,roc_auc,"ROC Curve - Random Forest")

