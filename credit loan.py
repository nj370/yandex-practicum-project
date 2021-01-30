#!/usr/bin/env python
# coding: utf-8


# Please see for details: https://www.markdowntutorial.com/
# </div>

# ### Step 1. Open the data file and have a look at the general information. 

# In[1]:


#import libraries that are needed to complete this project
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemma = WordNetLemmatizer()


#read the file
df=pd.read_csv('/datasets/credit_scoring_eng.csv')

#open the dataframe to investigate the data before preprocessing the data.
df.head(50)

print(len(df))

#general information about the data
print(df.info())
print(df.describe())


# ### Conclusion 
# 
# 

# many days employed values are negative. Such values are probably errors, and should be corrected.On closer inspection, I noticed that days employed values of samples that are classified as retired are positive, whilst the days employed values of samples that are classified as employed are negative

# <div class="alert alert-danger" role="alert">

# </div>

# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 2:
#     
# Well done that you described problems with data.
# </div>

# ### Step 2. Data preprocessing

# ### Processing missing values

# In[2]:


# turn days_employed values into positive values
df['days_employed']=abs(df['days_employed'])
#check the transformation
print(df.head(10))
#find missing values
print(df.isnull().sum())

#null values are present in quantitative variable columns only
#find the mean and median of each variable to assess the skewness of the data
print("days employed mean",df['days_employed'].mean())
print("income mean: ",df['total_income'].mean())
print("days employed median",df['days_employed'].median())
print("total income median",df['total_income'].median())

#replace null values with the statistics calculated
df['days_employed'].fillna(df['days_employed'].median(),inplace=True)
df['total_income'].fillna(df['total_income'].median(),inplace=True)

#check
print(df.isnull().sum())




# ### Conclusion

# the days employed values are corrected by being transformed into positve values. I checked for missing values and replaced the identified missing values.Columns that have missing values are days_employed and total_income. The numbers of missing values for days_employed and total_income are equal, this implies that the type of missing values in this study is missing at random.I hypothesize that the interconnection between the two are caused by samples not being able to remember how many days they have been employed, hence not being able to estimate total incomes.

# <div class="alert alert-success" role="alert">

# </div>

# ### Data type replacement

# In[3]:


#investigate data type
print(df.info())

#transform real data into integers
df['days_employed']=df['days_employed'].astype('int64')
df['total_income']=df['total_income'].astype('int64')

#check
df.info()


# ### Conclusion

# I decided to convert values that are of float type to values that are of integer type. This is because the decimal points are not of huge importance and ,therefore, can be ignored.

# <div class="alert alert-success" role="alert">

# </div>

# ### Processing duplicates

# In[4]:


#look for duplicates
print(df['education'].value_counts())
print(df['family_status'].value_counts())
print(df['gender'].value_counts())
print(df['income_type'].value_counts())
print(df['purpose'].value_counts())

#fix duplication issues in education column
df['education']=df['education'].str.lower()
print(df['education'].value_counts())

#create dict
purpose_dict={"wedding ceremony": "wedding",                          
"having a wedding": "wedding",                        
"to have a wedding": "wedding",                          
"real estate transactions": "real estate",                   
"buy commercial real estate": "real estate",                 
"housing transactions": "real estate",                        
"buying property for renting out" : "real estate",            
"transactions with commercial real estate" : "real estate",    
"purchase of the house" : "real estate",                       	
"housing" : "real estate" ,                                    
"purchase of the house for my family" : "real estate",       
"construction of own property" : "real estate",               
"property" : "real estate",                                   
"transactions with my real estate" : "real estate",          
"building a real estate" : "real estate",                   
"buy real estate" : "real estate",                           
"building a property" :"real estate" ,                        
"purchase of my own house" : "real estate" ,                  
"housing renovation" : "house renovation",                          
"buy residential real estate" :"real estate",                 
"buying my own car" : "auto loan" ,                         
"going to university" : "education" ,                        
"car" : "auto loan" ,                                        
"second-hand car purchase" : "auto loan" ,                   
"to own a car" : "auto loan",                              
"buying a second-hand car" : "auto loan",                    
"cars" : "auto loan",                                       
"to buy a car" : "auto loan",                                
"supplementary education" : "education" ,                  
"car purchase" : "auto loan",                               
"purchase of a car" : "auto loan",                           
"university education" : "education",                        
"to get a supplementary education" : "education" ,         
"education" : "education",                                   
"getting an education" : "education",                        
"profile education" : "education" ,                          
"getting higher education" : "education",                    
"to become educated" : "education",                          
}
#fix duplication issue in purpose column
df['purpose'] = df['purpose'].map(purpose_dict)
print(df.head(10))

#check for anomaly
#children column has anomalies.
df['children']=abs(df['children'])
df.loc[df['children']==20,'children']=2
#recheck
df.describe()
#anomaly in days employed
df.loc[df['days_employed']==401755,'days_employed']=2194
df.loc[df['dob_years']==0,'dob_years']=42


# ### Conclusion

# Quantitave columns with duplicates are ignored since it is normal for quantitave columns to have several duplicates. For example, many people have the same numbers of kids or are of the same age.I examined categorical columns to find duplicates. The columns with duplicates are 'purpose' and 'education'. One of the possible reasons why  there are dupicates in the purpose column and education column is that the the bank did not format the questions as  close-ended questions.As a result, open-ended questions result in the bank getting responses that have the same meaning but do not come in the form of preset answers. Duplicates were not replaced but manipulated for further analysis. duplicates in the education column can be easily manipulated by converting all values to lower case. Duplicates in the purpose column is more problematic and cannot simply be dropped due to pyhton not being able to detect the similarity of the contexts of the purposes. I created a dictionary and group the purposes into 5 groups, which are auto loan, real estate, education, wedding and house renovation.I mapped the dictionary to replace the duplicates.
# the anomalies I found are in the children column and days_employed column and dob_year. the anomalies in the chldren column are values of -1 and 20. The values are probably caused by human errors. I fixed the issues by transforming values into positive values and change data points with values of 20 to 2.
# 
# The anomaly in the the days employed is a value of 401755. It is probably a type so I changed it to the mdeian value of 2194 instead. DOB_year column also contains an anomaly value of 0.I changed it to a median value of 42 instead.
# 
# 

# <div class="alert alert-success" role="alert">

# </div>

# <div class="alert alert-danger" role="alert">
#
# </div>

# <div class="alert alert-success" role="alert">

# </div>

# ### Categorizing Data

# In[5]:


#categorising by education
education_dict=df[['education', 'education_id']]
education_dict=education_dict.drop_duplicates().reset_index(drop=True)
print(education_dict)

#categorising by family status
family_status_dict=df[['family_status','family_status_id']]
family_status_dict=family_status_dict.drop_duplicates().reset_index(drop=True)
print(family_status_dict)



print(df.head(10))

#categorising by income
#use quartile and median to categorise income groups
# categorise incomes into 3 groups- low income,middle income and high income.
income_median=df['total_income'].median()
highincome=df['total_income'].quantile(0.75)
lowincome=df['total_income'].quantile(0.25)

def income_categorisation(income):
    if income > highincome:
        return 'high income'
    if income < lowincome:
        return 'low income'
    else:
        return 'middle income'
    
df['income group']= df['total_income'].apply(income_categorisation)
print(df.head())

#categorising default
def default_or_not(default):
    if default == 1:
        return "has defaulted"
    if default == 0:
        return "has never defaulted"

df["polar_question default"]= df['debt'].apply(default_or_not)
print(df.head())


# ### Conclusion

# in order to answer the three questions required by this assignment, I categorised data by people with kids, income and marital status. 

# <div class="alert alert-danger" role="alert">

# </div>

# <div class="alert alert-success" role="alert">

# </div>

# ### Step 3. Answer these questions

# - Is there a relation between having kids and repaying a loan on time?

# In[6]:


#count the combinations
q0=df.groupby(['children','polar_question default'])['polar_question default'].count()
print(q0)
#get probability of no children defaulting
nochildrend=q0.iloc[0]
nochildrennd=q0.iloc[1]
nochildrenpd=(nochildrend/(nochildrend+nochildrennd))
print('probability of people with no children defaulting {:.2%}'.format(nochildrenpd))
#get probability of 1 child defaulting
onechildrend=q0.iloc[2]
onechildrennd=q0.iloc[3]
onechildrenpd=(onechildrend/(onechildrend+onechildrennd))
print('probability of people with 1 children defaulting {:.2%}'.format(onechildrenpd))
twochildrend=q0.iloc[4]
twochildrennd=q0.iloc[5]
twochildrenpd=(twochildrend/(twochildrend+twochildrennd))
print('probability of people with 2 children defaulting {:.2%}'.format(twochildrenpd))
threechildrend=q0.iloc[6]
threechildrennd=q0.iloc[7]
threechildrenpd=(threechildrend/(threechildrend+threechildrennd))
print('probability of people with 3 children defaulting {:.2%}'.format(threechildrenpd))
fourchildrend=q0.iloc[8]
fourchildrennd=q0.iloc[9]
fourchildrenpd=(fourchildrend/(fourchildrend+fourchildrennd))
print('probability of people with 4 children defaulting {:.2%}'.format(onechildrenpd))

print('probability of people with 5 children defaulting is 0%')


# ### Conclusion

# The data shows that probability of not having kids defaulting is lower than probabilities of people with kids defaulting.However, probability of poeple with 5 children defaulting is the lowest (0%). I hypothesise that this is because only wealthy people have 5 children.

# <div class="alert alert-success" role="alert">

# </div>

# - Is there a relation between marital status and repaying a loan on time?

# In[7]:


#count the combination
q1=df.groupby(['family_status','polar_question default'])['polar_question default'].count()
print(q1)
#get probability of civil partnership defaulting
civild=q1.iloc[0]
civilnd=q1.iloc[1]
civilpd=(civild/(civild+civilnd))
print('probability of civil partnership defaulting {:.2%}'.format(civilpd))
#get probability divorced defaulting
divorcedd=q1.iloc[2]
divorcednd=q1.iloc[3]
divorcedpd=(divorcedd/(divorcedd+divorcednd))
print('probability of divorced defaulting {:.2%}'.format(divorcedpd))
#get probability of married defaulting
marriedd=q1.iloc[4]
marriednd=q1.iloc[5]
marriedpd=(marriedd/(marriedd+marriednd))
print('probability of married defaulting {:.2%}'.format(marriedpd))
#get probability of unmarried defaulting
unmarriedd=q1.iloc[6]
unmarriednd=q1.iloc[7]
unmarriedpd=(unmarriedd/(unmarriedd+unmarriednd))
print('probability of unmarried defaulting {:.2%}'.format(unmarriedpd))
#get probability of widow defaulting
widowd=q1.iloc[8]
widownd=q1.iloc[9]
widowpd=(widowd/(widowd+widownd))
print('probability of widow/widower {:.2%}'.format(widowpd))


# ### Conclusion

# unmarried has the highest default rate among the marital statuses.It is very possible that being in partnership with someone is more economical than living alone. people who are in a partnership may have high incomes by combining incomes whilst benefiting from economy of scale such as bulk purchasing.Civil partnership has the second highest default rate. I hypothesise that this is because combining incomes may be difficult or may not be possible due to a lack of law to support such kind of arrangement. The 3 other marital statuses have very similar rates. Married people have lower default rate due to the explanation given above. The other 2 martial statuses are the lowest possibly because of the same reason. I hypothesise that these people have either inherited assets from their deceased partners or gain alimony after the divorce.

# <div class="alert alert-success" role="alert">

# </div>

# - Is there a relation between income level and repaying a loan on time?

# In[8]:


# count combinations
q2=df.groupby(['income group','polar_question default'])['polar_question default'].count()
print(q2)
#get probability of high incomers defaulting
highincomed=q2.iloc[0]
highincomend=q2.iloc[1]
highincomepd=(highincomed/(highincomed+highincomend))
print('probability of high incomer defaulting {:.2%}'.format(highincomepd))
#get probability of low incomers defaulting
lowincomed=q2.iloc[2]
lowincomend=q2.iloc[3]
lowincomepd=(lowincomed/(lowincomed+lowincomend))
print('probability of low incomer defaulting {:.2%}'.format(lowincomepd))
#get probability of middle incomers defaulting
middleincomed=q2.iloc[4]
middleincomend=q2.iloc[5]
middleincomepd=(middleincomed/(middleincomed+middleincomend))
print('probability of middle incomer defaulting {:.2%}'.format(middleincomepd))


# ### Conclusion

# the data confirms the presumption that high incomers have lower default rates. The data shows that low incomers  have lower defualt rate than that of middle incomers. this is possibly because many low incomers may not be able to take out loans due to their limited incomes, hence they have lower default rate.

# - How do different loan purposes affect on-time repayment of the loan?

# In[9]:


# count combinations
q3=df.groupby(['purpose','polar_question default'])['polar_question default'].count()
print(q3)
#get probability of auto loan defaulting
autod=q3.iloc[0]
autond=q3.iloc[1]
autopd=(autod/(autod+autond))
print('probability of auto loan defaulting {:.2%}'.format(autopd))
#get probability of education loan defaulting
educationd=q3.iloc[2]
educationnd=q3.iloc[3]
educationpd=(educationd/(educationd+educationnd))
print('probability of education defaulting {:.2%}'.format(educationpd))
#get probability of renovation defaulting
renovationd=q3.iloc[4]
renovationnd=q3.iloc[5]
renovationpd=(renovationd/(renovationd+renovationnd))
print('probability of house renovation defaulting {:.2%}'.format(renovationpd))
#get probability of real estate defaulting
realestated=q3.iloc[6]
realestatend=q3.iloc[7]
realestatepd=(realestated/(realestated+realestatend))
print('probability of real estate defaulting {:.2%}'.format(realestatepd))
#get probability of wedding defaulting
weddingd=q3.iloc[8]
weddingnd=q3.iloc[9]
weddingpd=(weddingd/(weddingd+weddingnd))
print('probability of wedding defaulting {:.2%}'.format(weddingpd))



# <div class="alert alert-success" role="alert">

# </div>

# ### Conclusion

# the probabilities of auto loan and education defaulting are the highest. This is probably because the payments on these loans take up a huge proportion of a person's income. Although payments on real estate also take up a huge proportion of a person's income, real estates generate cashflows for holders, hence lower default rate. default rate of house renovation is the lowest probably because payment on the loans take up a small proportion of a person's income. 

# 

# ### Step 4. General conclusion

# although conclusions have been given for each stage, readers should be aware that statistical tests such as logistic regression and goodness-of-fit tests should be conducted before any conclusions can be made. Future studies should run logistic regressions to test the hypotheses posed in this exercise. Logistic regression is appropriate for this type of question and allow for the effects of different variables to be controlled.
# 
# the characters of the best clients are:
# have no children or 5 children
# are divorced,widower or married
# have high incomes ( higher than > 32549)
# 
# 

# <div class="alert alert-danger" role="alert">

# </div>

# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 2:

# </div>

# ### Project Readiness Checklist
# 
# Put 'x' in the completed points. Then press Shift + Enter.

# - [x]  file open;
# - [x]  file examined;
# - [x]  missing values defined;
# - [x]  missing values are filled;
# - [x]  an explanation of which missing value types were detected;
# - [x]  explanation for the possible causes of missing values;
# - [x]  an explanation of how the blanks are filled;
# - [x]  replaced the real data type with an integer;
# - [x]  an explanation of which method is used to change the data type and why;
# - [x]  duplicates deleted;
# - [x]  an explanation of which method is used to find and remove duplicates;
# - [x]  description of the possible reasons for the appearance of duplicates in the data;
# - [x]  data is categorized;
# - [x]  an explanation of the principle of data categorization;
# - [x]  an answer to the question "Is there a relation between having kids and repaying a loan on time?";
# - [x]  an answer to the question " Is there a relation between marital status and repaying a loan on time?";
# - [x]   an answer to the question " Is there a relation between income level and repaying a loan on time?";
# - [x]  an answer to the question " How do different loan purposes affect on-time repayment of the loan?"
# - [x]  conclusions are present on each stage;
# - [x]  a general conclusion is made.

# In[ ]:




