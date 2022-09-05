import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
import random
import plotly.express as px
import plotly 
plotly.offline.init_notebook_mode() 
#get_ipython().system('pwd')

completed_session = pd.read_csv(path)
abandoned_session = pd.read_csv(path)

#Basic information about the two dataframes
print('Completed Form Sessions')
completed_session.info()
print('Abandoned Form Sessions')
abandoned_session.info()

#Missing Values
sns.heatmap(completed_session.isnull(),cmap="YlGnBu")
Cmissing_values = completed_session.isnull().sum()
Cmissing_values

sns.heatmap(abandoned_session.isnull(),cmap="YlGnBu")
Amissing_values = abandoned_session.isnull().sum()
Amissing_values

#Dropping columns 
completed_session = completed_session.drop(columns = ['domain','attributes/consumerName','organisationUuid','attributes/formDefinitionId','attributes/onlineEntryMethodCode','timeStamp', 'attributes/journeyType','attributes/doubleClickId','formSlug','formUuid'])
abandoned_session = abandoned_session.drop(columns = {'domain','attributes/consumerName','organisationUuid','attributes/formDefinitionId','attributes/onlineEntryMethodCode','timeStamp', 'attributes/journeyType','attributes/doubleClickId','formSlug','formUuid'})

# Correcting missing values
completed_session.loc[completed_session['field/tagName'] == 'SELECT', 'field/type'] = 'dropdown'
abandoned_session.loc[abandoned_session['field/tagName'] == 'SELECT', 'field/type'] = 'dropdown'
completed_session.loc[completed_session['field/id'] == 'apply-screen:keeping-in-touch:future-soft-search-permission', 'field/name'] = 'future-soft-search-permission'
abandoned_session.loc[abandoned_session['field/id'] == 'apply-screen:keeping-in-touch:future-soft-search-permission', 'field/name'] = 'future-soft-search-permission'
completed_session.loc[completed_session['field/id'] == 'apply-screen:keeping-in-touch:marketing-permission', 'field/name'] = 'marketing-permission'
abandoned_session.loc[abandoned_session['field/id'] == 'apply-screen:keeping-in-touch:marketing-permission', 'field/name'] = 'marketing-permission'

#Dropping inconsistent field names
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

question_sequence = {'title':1, 'first-name' : 2,'last-name' : 3,'date-of-birth-day' : 4, 'date-of-birth-month' : 5, 'date-of-birth-year' : 6,'email-address' : 7,
                     'mobile-number' : 8,  'home-number' : 9, 'employment-status' : 10, 'occupation' : 11, 'yearly-income' : 12, 'other-household-income' : 13, 'number-of-dependants' : 14,'cash-advance' : 15,
                     'residential-status' : 16,  'rent-mortgage-payments' : 17,'current-address-lookup-postcode' : 18, 'current-address-lookup-house-number' : 19,'house-number' : 20, 'house-name' : 21, 
                     'flat-number' : 22,'street-name' : 23, 'address-line-2' : 24,'town-city' : 25,'county' : 26,'postcode' : 27, 'years-at-address' : 28,'previous-address-lookup-postcode' : 29,
                     'previous-address-lookup-house-number' : 30, 'marketing-permission':31, 'future-soft-search-permission':32}


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

# Get field names
def get_fieldname(field):
    return field.split(':')[-1]

completed_session['field_name'] = completed_session['field/name'].dropna().apply(get_fieldname)
abandoned_session['field_name'] = abandoned_session['field/name'].dropna().apply(get_fieldname)

# Grouping the questions according to their steps
def get_fieldstep(field):
    group = field.split(':')[0]
    field_name = field.split(':')[-1]
    if field_name in ['bt-amount', 'interested-in-bt', 'title','date-of-birth-day','date-of-birth-month','date-of-birth-year', 'email-address', 'first-name', 'last-name']:
        return 'Step1_LandingPage'
    if field_name in ['date-of-birth-day', 'date-of-birth-month', 'date-of-birth-year', 'email-address', 'home-number', 'mobile-number']:
        return 'Step2_AboutYou'
    if field_name in ['employment-status', 'occupation', 'yearly-income','other-household-income']:
        return 'Step3_Income'
    if field_name in ['number-of-dependants', 'cash-advance', 'rent-mortgage-payments', 'residential-status']:
        return 'Step4_Outgoing'
    if field_name in ['address-line-2', 'county', 'flat-number', 'house-name', 'house-number', 'list', 'postcode', 'street-name', 'town-city', 'years-at-address', 'address-line-2', 'county', 'flat-number', 'house-name', 'house-number', 'list', 'postcode', 'street-name', 'town-city', 'years-at-address','current-address-lookup-house-number','current-address-lookup-postcode','previous-address-lookup-house-number','previous-address-lookup-postcode']:
        return 'Step5_Address'
    if field_name in ['marketing-permission','future-soft-search-permission']:
        return 'Step6_Permission'   
completed_session['field_step'] = completed_session['field/name'].dropna().apply(get_fieldstep)
abandoned_session['field_step'] = abandoned_session['field/name'].dropna().apply(get_fieldstep)
unique_form_step = ['Step1_LandingPage','Step2_AboutYou','Step3_Income','Step4_Outgoing','Step5_Address','Step6_Permissions']
no_of_steps = 6
unique_steps =  pd.unique(completed_session['field_step'].dropna())

total_visitors_at_each_step = abandoned_session.groupby('field_step')['visitorId'].nunique()
total_visitors_at_each_step

# Unique field names in both sessiosn
unique_field_name_completed = pd.unique(completed_session['field_name'])
unique_field_name_completed = pd.Series(unique_field_name_completed).dropna()
count_unique_field_name_completed = len(unique_field_name_completed)
unique_field_name_abandoned = pd.unique(abandoned_session['field_name'])
unique_field_name_abandoned = pd.Series(unique_field_name_abandoned).dropna()
count_unique_field_name_abandoned = len(unique_field_name_abandoned)
print("Total number of question in completed sessions:",count_unique_field_name_completed )
print("Total number of question in abandoned sessions:", count_unique_field_name_abandoned)

# Field types in both sessions
unique_field_type_completed = pd.unique(completed_session['field/type'])
unique_field_type_completed = pd.Series(unique_field_type_completed).dropna()
count_unique_field_type_completed = len(unique_field_type_completed)
unique_field_type_abandoned = pd.unique(abandoned_session['field/type'])
unique_field_type_abandoned = pd.Series(unique_field_type_abandoned).dropna()
count_unique_field_type_abandoned = len(unique_field_type_abandoned)
print(unique_field_type_completed,unique_field_type_abandoned)

# Selecting Sample of 5000
selected_size = 5000
unique_users_completed = unique_users_completed[:selected_size]
count_users_completed = selected_size
unique_users_abandoned = unique_users_abandoned[:selected_size]
count_users_abandoned = selected_size

# Stats of questions attempted in abandoned forms
questions_attempted_per_abandoned_form = []
for visitor in unique_users_abandoned:
    interaction = abandoned_session.loc[(abandoned_session['visitorId'] == visitor)]
    questions_attempted = len(pd.unique(interaction['field_name']))
    questions_attempted_per_abandoned_form.append(questions_attempted)

print("Statistics of questions attempted per abandoned form")
print(f"Mean = {statistics.mean(questions_attempted_per_abandoned_form)}")
print(f"Median = {statistics.median(questions_attempted_per_abandoned_form)}")
print(f"Mode = {statistics.mode(questions_attempted_per_abandoned_form)}")

# Time spent from arival time 
# Completed Sessions
completed_session['time_spent_unix'] = np.nan
for visitor in unique_users_completed:
  prev=-1
  form_completed = completed_session.loc[completed_session['visitorId'] == visitor]
  form_completed = form_completed.sort_values('receivedAt')
  for index, row in form_completed.iterrows():
    if prev == -1:
      time_spent_unix = 0  
    else:
      time_spent_unix = round((row['receivedAt']-form_completed.loc[prev,('receivedAt')])/1000,2)
    prev = index
    completed_session.loc[index,('time_spent_unix')] = time_spent_unix

# Abandoned Sessions

abandoned_session['time_spent_unix'] = np.nan
for visitor in unique_users_abandoned:
  prev=-1
  form_abandoned = abandoned_session.loc[abandoned_session['visitorId'] == visitor]
  form_abandoned = form_abandoned.sort_values('receivedAt')
  for index, row in form_abandoned.iterrows():
    if prev == -1:
      time_spent_unix = 0
    else:
      time_spent_unix = round((row['receivedAt']-form_abandoned.loc[prev,('receivedAt')])/1000,2)
    prev = index
    abandoned_session.loc[index,('time_spent_unix')] = time_spent_unix

# Calculate time spent per form
time_spent_unix_per_form = pd.DataFrame()
time_spent_unix = []
for visitor in unique_users_completed:
  form_completed = completed_session.loc[completed_session['visitorId'] == visitor]
  time_spent_unix.append(form_completed['time_spent_unix'].sum())
time_spent_unix_per_form['Completed'] = time_spent_unix

time_spent_unix = []
for visitor in unique_users_abandoned:
  form_abandoned = abandoned_session.loc[abandoned_session['visitorId'] == visitor]
  time_spent_unix.append(form_abandoned['time_spent_unix'].sum())
time_spent_unix_per_form['Abandoned'] = time_spent_unix

average_time_spent_unix_completed = round(time_spent_unix_per_form['Completed'].sum()/count_users_completed,2)
average_time_spent_unix_abandoned = round(time_spent_unix_per_form['Abandoned'].sum()/count_users_abandoned,2)
print("Average time for completed sessions is", average_time_spent_unix_completed)
print("Average time for abandoned sessions is", average_time_spent_unix_abandoned)

#Graph - Average Time Taken per Visitor 
colour = {'Completed':'dodgerblue','Abandoned':'orangered'}
ax = sns.violinplot(data=time_spent_unix_per_form, palette=colour)
ax.set(title='Total Time Spent per Session by a Visitor')
ax.set_xlabel('Session Type')
ax.set_ylabel('Total Time (Unix)')
plt.show()

figure = time_spent_unix_per_form.plot(kind='box', title='Total Time Spent per Session by Visitors', showfliers = False)
figure.set_xlabel('Session Type')
figure.set_ylabel('Total Time (Seconds)')
plt.show()

# Time spent per question
# Completed Sessions
time_completed_per_q_dt = pd.DataFrame()
for question in unique_field_name_completed:
  rows = completed_session.loc[(completed_session['field_name'] == question) & (completed_session['time_spent_unix'] != np.nan)]
  time_spent_unix = []
  for visitor in unique_users_completed:
    time_spent_unix.append(rows.loc[rows['visitorId'] == visitor]['time_spent_unix'].sum())  
  time_completed_per_q_dt[question] = time_spent_unix
# Abandoned Sessions
time_abandoned_per_q_dt = pd.DataFrame()
for question in unique_field_name_abandoned:
  rows = abandoned_session.loc[(abandoned_session['field_name'] == question) & (abandoned_session['time_spent_unix'] != np.nan)]
  time_spent_unix = []
  for visitor in unique_users_abandoned:
    time_spent_unix.append(rows.loc[rows['visitorId'] == visitor]['time_spent_unix'].sum())  
  time_abandoned_per_q_dt[question] = time_spent_unix

#Graph - Average Time per Question 
df_completed = pd.DataFrame({"Question names":time_completed_per_q_dt.sum(axis=0).index,
                   "Avg time":(time_completed_per_q_dt.sum(axis=0).values/count_users_completed).round(2),
                   "Session":"Completed"})
df_abandoned = pd.DataFrame({"Question names":time_abandoned_per_q_dt.sum(axis=0).index,
                   "Avg time":(time_abandoned_per_q_dt.sum(axis=0).values/count_users_abandoned).round(2),
                   "Session":"Abandoned"})
df = pd.concat([df_completed,df_abandoned], axis=0)
fig = px.bar(df, x="Question names", y="Avg time",
             color='Session', barmode='group',height=700)
fig.update_layout(
    title={'text': "Average Visitor Time per Field",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Field Name",yaxis_title="Average Time (seconds)")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.show()

#Graph - Average Time Taken per Visitor 
df_ordered= time_completed_per_q_dt[['title', 'first-name','last-name','date-of-birth-day', 'date-of-birth-month', 'date-of-birth-year','email-address',
                     'mobile-number',  'home-number', 'employment-status', 'occupation', 'yearly-income', 'other-household-income', 'number-of-dependants','cash-advance',
                     'residential-status',  'rent-mortgage-payments','current-address-lookup-postcode', 'current-address-lookup-house-number','list','house-number', 'house-name', 
                     'flat-number','street-name', 'address-line-2','town-city','county','postcode', 'years-at-address','previous-address-lookup-postcode',
                     'previous-address-lookup-house-number', 'marketing-permission', 'future-soft-search-permission']]
figure = df_ordered.plot(kind='box', title='Time Taken per Field - Completed Sessions (Outliers Removed)', showfliers=False,)
plt.xticks(rotation = 90)
figure.set_xlabel('Field Name')
figure.set_ylabel('Average Time (seconds)')
df_ordered= time_abandoned_per_q_dt[['title', 'first-name','last-name','date-of-birth-day', 'date-of-birth-month', 'date-of-birth-year','email-address',
                     'mobile-number',  'home-number', 'employment-status', 'occupation', 'yearly-income', 'other-household-income', 'number-of-dependants','cash-advance',
                     'residential-status',  'rent-mortgage-payments','current-address-lookup-postcode', 'current-address-lookup-house-number','list','house-number', 'house-name', 
                     'flat-number','street-name', 'address-line-2','town-city','county','postcode', 'years-at-address','previous-address-lookup-postcode',
                     'previous-address-lookup-house-number', 'marketing-permission', 'future-soft-search-permission']]
figure = df_ordered.plot(kind='box', title='Time Taken per Field  - Abandoned Sessions (Outliers Removed)',showfliers=False)
plt.xticks(rotation = 90)
figure.set_xlabel('Field Name')
figure.set_ylabel('Average Time (seconds)')
plt.show()

# Time Spent per Field Type
# Completed Sessions
time_completed_per_type_dt = pd.DataFrame()
for tpe in unique_field_type_completed:
  rows = completed_session.loc[(completed_session['field/type'] == tpe) & (completed_session['time_spent_unix'] != np.nan)]
  time_spent_unix = []
  for visitor in unique_users_completed:
    time_spent_unix.append(rows.loc[rows['visitorId'] == visitor]['time_spent_unix'].sum())  
  time_completed_per_type_dt[tpe] = time_spent_unix
# Abandoned Sessions
time_abandoned_per_type_dt = pd.DataFrame()
for tpe in unique_field_type_abandoned:
  rows = abandoned_session.loc[(abandoned_session['field/type'] == tpe) & (abandoned_session['time_spent_unix'] != np.nan)]
  time_spent_unix = []
  for visitor in unique_users_abandoned:
    time_spent_unix.append(rows.loc[rows['visitorId'] == visitor]['time_spent_unix'].sum())  
  time_abandoned_per_type_dt[tpe] = time_spent_unix

#Graph - Average time per Field Type
df_completed = pd.DataFrame({"Field Type":time_completed_per_type_dt.sum(axis=0).index,
                   "Avg Time":(time_completed_per_type_dt.sum(axis=0).values/count_users_completed).round(2),
                   "Session":"Completed"})
df_abandoned = pd.DataFrame({"Field Type":time_abandoned_per_type_dt.sum(axis=0).index,
                   "Avg Time":(time_abandoned_per_type_dt.sum(axis=0).values/count_users_abandoned).round(2),
                   "Session":"Abandoned"})
df = pd.concat([df_completed,df_abandoned], axis=0)
fig = px.bar(df, x="Field Type", y="Avg Time",color='Session', barmode='group',height=700)
fig.update_layout(title={'text': "Average Time on Every Field Type",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Field Type",yaxis_title="Average Time (Unix timein seconds)")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.show()

#Graph - Time Taken per Field Type - box plots  
figure = time_completed_per_type_dt.plot(kind='box', title='Time Taken Per Field Type - Completed Sessions (Outliers Removed)', showfliers=False)
plt.xticks(rotation = 90)
figure.set_xlabel('Field Type')
figure.set_ylabel('Time (seconds)')
df_ordered= time_abandoned_per_type_dt[['tel','submit','dropdown','text','radio','email','button','checkbox']]
figure = df_ordered.plot(kind='box', title='Time Taken Per Field Type - Abandoned Sessions (Outliers Removed)', showfliers=False)
plt.xticks(rotation = 90)
figure.set_xlabel('Field Type')
figure.set_ylabel('Time (seconds)')
plt.show()

# Time per Form Step
# Completed
time_completed_per_step = pd.DataFrame()
for step in unique_form_step:
  rows = completed_session.loc[(completed_session['field_step'] == step) & (completed_session['time_spent_unix'] != np.nan)]
  time_spent_unix = []
  for visitor in unique_users_completed:
    time_spent_unix.append(rows.loc[rows['visitorId'] == visitor]['time_spent_unix'].sum())  
  time_completed_per_step[step] = time_spent_unix

# Abandoned
time_abandoned_per_step = pd.DataFrame()
for step in unique_form_step:
  rows = abandoned_session.loc[(abandoned_session['field_step'] == step) & (abandoned_session['time_spent_unix'] != np.nan)]
  time_spent_unix = []
  for visitor in unique_users_abandoned:
    time_spent_unix.append(rows.loc[rows['visitorId'] == visitor]['time_spent_unix'].sum())  
  time_abandoned_per_step[step] = time_spent_unix

# Graph - Average time per form step
df_completed = pd.DataFrame({"Field step":time_completed_per_step.sum(axis=0).index,
                   "Average time taken":(time_completed_per_step.sum(axis=0).values/count_users_completed).round(2),
                   "Form":"Completed"})
df_abandoned = pd.DataFrame({"Field step":time_abandoned_per_step.sum(axis=0).index,
                   "Average time taken":(time_abandoned_per_step.sum(axis=0).values/count_users_abandoned).round(2),
                   "Form":"Abandoned"})
df = pd.concat([df_completed,df_abandoned], axis=0)
fig = px.bar(df, x="Field step", y="Average time taken", color='Form', barmode='group',height=700)
fig.update_layout(title={'text': "Average Time - Form Steps",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Form Step",yaxis_title="Average Time (Unix time)")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.show()

#Graph - Average Time per form step
fig = time_completed_per_step.plot(kind='box', title='Time Taken on Form Steps - Completed Sessions (Outliers Removed)', showfliers=False)
plt.xticks(rotation = 90)
fig.set_xlabel('Form Step')
fig.set_ylabel('Time (Seconds)')
fig = time_abandoned_per_step.plot(kind='box', title='Time Taken on Form Steps - Abandoned Sessions (Outliers Removed)', showfliers=False)
plt.xticks(rotation = 90)
fig.set_xlabel('Form Step')
fig.set_ylabel('Time (Seconds)')

plt.show()

# Backspaces per question
# Completed Sessions 
total_backspaces_completed = dict()
for question in unique_field_name_completed:
  rows = completed_session.loc[(completed_session['field_name'] == question) & (completed_session['key'] == 'Backspace')]
  total_backspaces_completed[question] = rows.size
# Abandoned Sessions 
total_backspaces_abandoned = dict()
for question in unique_field_name_abandoned:
  rows = abandoned_session.loc[(abandoned_session['field_name'] == question) & (abandoned_session['key'] == 'Backspace')]
  total_backspaces_abandoned[question] = rows.size

# Graph - Average Backspaces per Question
df_completed = pd.DataFrame({"Question":total_backspaces_completed.keys(),
                   "Average Backspaces":(list(total_backspaces_completed.values())/np.int64(pd.unique(completed_session['visitorId']).size)).round(2),
                   "Session":"Completed"})
df_abandoned = pd.DataFrame({"Question":total_backspaces_abandoned.keys(),
                   "Average Backspaces":(list(total_backspaces_abandoned.values())/np.int64(pd.unique(abandoned_session['visitorId']).size)).round(2),
                   "Session":"Abandoned"})
df = pd.concat([df_completed,df_abandoned], axis=0)
fig = px.bar(df, x="Question", y="Average Backspaces",
             color='Session', barmode='group',
             height=700)
fig.update_layout(title={'text': "Average Backspaces per Question",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Field Name",yaxis_title="Number of Backspaces")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.update_xaxes(categoryorder='total ascending')
fig.update_xaxes(categoryorder='array', categoryarray= ['title', 'first-name','last-name','date-of-birth-day', 'date-of-birth-month', 'date-of-birth-year','email-address',
                     'mobile-number',  'home-number', 'employment-status', 'occupation', 'yearly-income', 'other-household-income', 'number-of-dependants','cash-advance',
                     'residential-status',  'rent-mortgage-payments','current-address-lookup-postcode', 'current-address-lookup-house-number','list','house-number', 'house-name', 
                     'flat-number','street-name', 'address-line-2','town-city','county','postcode', 'years-at-address','previous-address-lookup-postcode',
                     'previous-address-lookup-house-number', 'marketing-permission', 'future-soft-search-permission'])
fig.show()

#Clicks per session
clicks_completed =  completed_session.loc[completed_session['type']=='click'].size
clicks_abandoned =  abandoned_session.loc[abandoned_session['type']=='click'].size
clicks_completed_per_user = int(clicks_completed/55550)
clicks_abandoned_per_user = int(clicks_abandoned/62428)
#Graph - Average Clicks per session
import plotly.express as px
df = pd.DataFrame({"Session":['Completed','Abandoned'], "Average Clicks per Session":[clicks_completed_per_user,clicks_abandoned_per_user]})
fig = px.bar(df, x="Session", y="Average Clicks per Session", text = "Average Clicks per Session", height=400)
fig.update_layout(title={'text': "Average Clicks Per Session",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Session Type",yaxis_title="Average Clicks")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.show()

#Clicks per question
# Completed
total_clicks_completed = dict()
for question in unique_field_name_completed:
  rows = completed_session.loc[(completed_session['field/name'] == question) & (completed_session['type'] == 'click')]
  total_clicks_completed[question] = rows.size

# Abandoned
total_clicks_abandoned = dict()
for question in unique_field_name_abandoned:
  rows = abandoned_session.loc[(abandoned_session['field/name'] == question) & (abandoned_session['type'] == 'click')]
  total_clicks_abandoned[question] = rows.size

# Proportion of Clicks per Field
df_completed = pd.DataFrame({"Field":total_clicks_completed.keys(),
                   "Average Clicks":(list(total_clicks_completed.values())/sum(total_clicks_completed.values())*100).round(2),
                   "Form":"Completed"})
df_abandoned = pd.DataFrame({"Field":total_clicks_abandoned.keys(),
                   "Average Clicks":(list(total_clicks_abandoned.values())/sum(total_clicks_abandoned.values())*100).round(2),
                   "Form":"Abandoned"})
df = pd.concat([df_completed,df_abandoned], axis=0)
fig = px.bar(df, x="Field", y="Average Clicks",
             color='Form', barmode='group',
             height=700)
fig.update_layout(title={'text': "Proportion of Clicks per Question",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Field Name",yaxis_title="Proportion of Clicks")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.update_xaxes(categoryorder='total ascending')
fig.update_xaxes(categoryorder='array', categoryarray= ['title', 'first-name','last-name','date-of-birth-day', 'date-of-birth-month', 'date-of-birth-year','email-address',
                     'mobile-number',  'home-number', 'employment-status', 'occupation', 'yearly-income', 'other-household-income', 'number-of-dependants','cash-advance',
                     'residential-status',  'rent-mortgage-payments','current-address-lookup-postcode', 'current-address-lookup-house-number','list','house-number', 'house-name', 
                     'flat-number','street-name', 'address-line-2','town-city','county','postcode', 'years-at-address','previous-address-lookup-postcode',
                     'previous-address-lookup-house-number', 'marketing-permission', 'future-soft-search-permission'])
fig.show()

# Last question user abandoned form at
questions_attempted_last_per_abandoned_form = []
for visitor in unique_users_abandoned:
    interaction = abandoned_session.loc[(abandoned_session['visitorId'] == visitor)]
    questions_attempted = pd.unique(interaction['field_name'].dropna())
    if questions_attempted.any():
        questions_attempted_last_per_abandoned_form.append(questions_attempted[-1])


unique_questions_abandoned_at = pd.unique(questions_attempted_last_per_abandoned_form)
freq_questions_abandoned_at = dict()
for question in unique_questions_abandoned_at:
    freq_questions_abandoned_at[question] = questions_attempted_last_per_abandoned_form.count(question)

print(list(freq_questions_abandoned_at))

df = pd.DataFrame({"Question":freq_questions_abandoned_at.keys(), "Number of forms abandoned at this question":freq_questions_abandoned_at.values()})
fig = px.bar(df, x="Question", y="Number of forms abandoned at this question", height=500)
fig.update_layout(title={'text': "Last Question for Abandoned Session Visitors",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Field Name",yaxis_title="Count")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.update_xaxes(categoryorder='total ascending')
fig.update_xaxes(categoryorder='array', categoryarray= ['title', 'first-name','last-name','date-of-birth-day', 'date-of-birth-month', 'date-of-birth-year','email-address',
                     'mobile-number',  'home-number', 'employment-status', 'occupation', 'yearly-income', 'other-household-income', 'number-of-dependants','cash-advance',
                     'residential-status',  'rent-mortgage-payments','current-address-lookup-postcode', 'current-address-lookup-house-number','list','house-number', 'house-name', 
                     'flat-number','street-name', 'address-line-2','town-city','county','postcode', 'years-at-address','previous-address-lookup-postcode',
                     'previous-address-lookup-house-number', 'marketing-permission', 'future-soft-search-permission'])
fig.show()

# Browser and device type
# Get browser name

def get_browser(user_agent):
    
    if "Chrome" in user_agent:
        return "Chrome"
    elif "Edg" in user_agent:
        return "Edge"
    elif "Firefox" in user_agent:
        return "Firefox"
    elif "OPR" in user_agent or "Opera" in user_agent:
        return "Opera"
    elif "Safari" in user_agent:
        return "Safari"
    elif "MSIE" in user_agent or "Trident" in user_agent:
        return "Internet Explorer"
    return "Other"    
completed_session['browser'] = completed_session['userAgent'].dropna().apply(get_browser)
abandoned_session['browser'] = abandoned_session['userAgent'].dropna().apply(get_browser)
print(f"Browser types in the Completed Sessions are: {pd.unique(completed_session['browser'].dropna())}")
print(f"Browser types in the Abandoned Sessions are: {pd.unique(abandoned_session['browser'].dropna())}")

# Get device type
devies = {1 : 'Mobile', 2 : 'Computer', 3 : 'Tablet'}
def get_device(user_agent):
    if "Mobile" in user_agent or "Mobi" in user_agent or "Android" in user_agent or "iPhone" in user_agent:
        return "Mobile"
    elif "Tablet" in user_agent:
        return "Tablet"       
    return "Computer"
completed_session['device'] = completed_session['userAgent'].dropna().apply(get_device)
abandoned_session['device'] = abandoned_session['userAgent'].dropna().apply(get_device)
print(f"Device types in the Completed Sessions are: {pd.unique(completed_session['device'].dropna())}")
print(f"Device types in the Abandoned Sessions are: {pd.unique(abandoned_session['device'].dropna())}")

keys = pd.unique(completed_session['browser'])
values = np.zeros(pd.unique(completed_session['browser']).size)
freq_browser_completed = dict(zip(keys,values))
for visitor in unique_users_completed:
    interaction = completed_session.loc[(completed_session['visitorId'] == visitor)]
    browser = pd.unique(interaction['browser'])[0]
    freq_browser_completed[browser] += 1 
print(f"Browser count for Completed Sessions is: {freq_browser_completed}")


keys = pd.unique(abandoned_session['browser'])
values = np.zeros(pd.unique(abandoned_session['browser']).size)
freq_browser_abandoned = dict(zip(keys,values))
for visitor in unique_users_abandoned:
    interaction = abandoned_session.loc[(abandoned_session['visitorId'] == visitor)]
    browser = pd.unique(interaction['browser'])[0]
    freq_browser_abandoned[browser] += 1 
print(f"Browser count for Abandoned Sessions is: {freq_browser_abandoned}")

keys = pd.unique(completed_session['device'])
values = np.zeros(pd.unique(completed_session['device']).size)
freq_device_completed = dict(zip(keys,values))
for visitor in unique_users_completed:
    interaction = completed_session.loc[(completed_session['visitorId'] == visitor)]
    device = pd.unique(interaction['device'])[0]
    freq_device_completed[device] += 1 
print(f"Device count for Completed Sessions is {freq_device_completed}")


keys = pd.unique(abandoned_session['device'])
values = np.zeros(pd.unique(abandoned_session['device']).size)
freq_device_abandoned = dict(zip(keys,values))
for visitor in unique_users_abandoned:
    interaction = abandoned_session.loc[(abandoned_session['visitorId'] == visitor)]
    device = pd.unique(interaction['device'])[0]
    freq_device_abandoned[device] += 1 
print(f"Device count for Abandoned Sessions is {freq_device_abandoned}")

# Checking if browser changed midway
completed_forms_with_changed_browser = 0
for visitor in unique_users_completed:
    interaction = completed_session.loc[(completed_session['visitorId'] == visitor)]
    if len(pd.unique(interaction['browser'].dropna())) > 1:
        completed_forms_with_changed_browser += 1
print(completed_forms_with_changed_browser)

abandoned_forms_with_changed_browser = 0
for visitor in unique_users_abandoned:
    interaction = abandoned_session.loc[(abandoned_session['visitorId'] == visitor)]
    if len(pd.unique(interaction['browser'].dropna())) > 1:
        abandoned_forms_with_changed_browser += 1
print(abandoned_forms_with_changed_browser)

print(f"Count of  browser change instance in completed forms = {completed_forms_with_changed_browser}")
print(f"Count of  browser change instance in abandoned forms = {abandoned_forms_with_changed_browser}")

# Step user abandoned form at
last_step_abandoned = []
for visitor in unique_users_abandoned:
    interaction = abandoned_session.loc[(abandoned_session['visitorId'] == visitor)]
    attempted_step = pd.unique(interaction['field_step'].dropna())
    if attempted_step.any():
        last_step_abandoned.append(attempted_step[-1])

unique_steps_abandoned = pd.unique(last_step_abandoned)
frequency = dict()
for step in unique_steps_abandoned:
    frequency[step] = last_step_abandoned.count(step)
print(frequency)

#Graph - Count Abandonment at each step
df = pd.DataFrame({"Step":frequency.keys(), "Abandoned form step":frequency.values()})
fig = px.bar(df, x="Step", y="Abandoned form step", text = "Abandoned form step", height=500)
fig.update_layout(title={'text': "Last Step Attempted before Abandonment",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Form Steps",yaxis_title="Count Abandoned")
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
fig.update_xaxes(categoryorder='category ascending')
fig.show()

def get_fieldNumber(field):
    if field in question_sequence:
        return question_sequence[field]
    
completed_session['field_number'] = completed_session['field_name'].dropna().apply(get_fieldNumber)
abandoned_session['field_number'] = abandoned_session['field_name'].dropna().apply(get_fieldNumber)

# Capturing Field Returns
# Completed Sessions 
completed_session['field_returns'] = np.nan
for visitor in unique_users_completed:
  prev=-1
  form_completed = completed_session.loc[completed_session['visitorId'] == visitor]
  form_completed = form_completed.sort_values('receivedAt')
  for index, row in form_completed.iterrows():
    if prev == -1:
      field_returns = 0
    else:
      field_returns = (row['field_number']-form_completed.loc[prev,('field_number')])
    prev = index
    completed_session.loc[index,('field_returns')] = field_returns

# Abandoned Sessions 
abandoned_session['field_returns'] = np.nan
for visitor in unique_users_abandoned:
  prev=-1
  form_abandoned = abandoned_session.loc[abandoned_session['visitorId'] == visitor]
  form_abandoned = form_abandoned.sort_values('receivedAt')
  for index, row in form_abandoned.iterrows():
    if prev == -1:
      field_returns = 0
    else:
      field_returns = (row['field_number']-form_abandoned.loc[prev,('field_number')])
    prev = index
    abandoned_session.loc[index,('field_returns')] = field_returns

questions_returned_at_completed = list(completed_session.loc[(completed_session['field_returns'] < 0)]['field_name'])
questions_returned_at_abandoned = list(abandoned_session.loc[(abandoned_session['field_returns'] < 0)]['field_name'])

# Getting Questions with Maximum Returns
import operator

unique_questions_returned_at_completed = pd.unique(questions_returned_at_completed)
freq_question_returned_at_completed = dict()
for question in unique_questions_returned_at_completed:
    freq_question_returned_at_completed[question] = questions_returned_at_completed.count(question)
print(freq_question_returned_at_completed)
print(f"Field with the maximum field returns in the completed sessions was: {max(freq_question_returned_at_completed.items(), key=operator.itemgetter(1))[0]}")


unique_questions_returned_at_abandoned = pd.unique(questions_returned_at_abandoned)
freq_question_returned_at_abandoned = dict()
for question in unique_questions_returned_at_abandoned:
    freq_question_returned_at_abandoned[question] = questions_returned_at_abandoned.count(question)
print(freq_question_returned_at_abandoned)
print(f"Field with the maximum field returns in the completed sessions was: {max(freq_question_returned_at_abandoned.items(), key=operator.itemgetter(1))[0]}")


