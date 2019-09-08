# This script takes in the DLS Student Dataset as distributed by the DBA
""" NOTES
2009-2018 Numerical & Discrete
2013-2018 Numerical & Discrete
2009-2018 Non-Duplicate Numerical & Discrete
2013-2018 Non-Duplicate Numerical & Discrete
"""

# Needed Import Statements
import numpy as np
import pandas as pd
# Importing the Raw Data
#dataset = pd.read_excel('RAW2008_to_2018_student_data.xlsx')
#discipline = pd.read_excel('RAW2008_to_2018_Discipline.xlsx')
dataset = pd.read_excel('4_3_2019_Data.xlsx')
discipline = pd.read_excel('4_3_2019_Discipline.xlsx')



# Fix the Hispanic Ethnic Column
# For STUDENT-ETHNIC, 3 used to denote hispanic, but after some time
# it was marked "Y" and another feature was added titlted "STUDENT-A-LATINO.
# The below looks at the new feature and changes STUDENT-ETHNIC to 3 if 'Y' is marked
dataset.loc[dataset['STUDENT-A-LATINO'] == 'Y', 'STUDENT-ETHNIC'] = '3'
dataset['STUDENT-ETHNIC'].value_counts()
dataset.loc[dataset['STUDENT-PROGRAM'] == 'G ', 'STUDENT-PROGRAM'] = 'LP'
dataset.loc[dataset['STUDENT-PROGRAM'] == ' ', 'STUDENT-PROGRAM'] = 'CP'
dataset.loc[dataset['STUDENT-PROGRAM'] == '  ', 'STUDENT-PROGRAM'] = 'CP'
dataset['STUDENT-PROGRAM'].value_counts()
# Merges discipline into Student DB
dataset = pd.merge(dataset,
                discipline,
                left_on = ['School Yr', 'STUDENT-ID'],
                right_on = ['School YR', 'Student ID'],
                how = 'left') 


# Dropping the individual grade columns
student_info = dataset.drop(dataset.columns[24:-1], axis = 1)
# Dropping rows where ID & row = NaN
student_info = student_info.dropna(how = 'all')

# Drops columns that we couldn't mass drop
drop_features = ['STUDENT-DATE-ENTERED',
                 'STUDENT-DATE-ENTERED',
                 'STUDENT-PARENT',
                 'STUDENT-PARENT-SALUTATION',
                 'STUDENT-PARENT-RELATIONSHIP',
                 'STUDENT-CRED-ERN',
                  'STUDENT-SIBLINGS', # This is dropped because all data says '00'
                  'STUDENT-WITHDRAW-DATE']
student_info.drop(drop_features, axis = 1, inplace = True)

# Changing School Year Data to int & Cleaning some Individual Rows
# Changing the 'School Yr' from ##-## to a single integer,
# which is the year they finished (i.e 17-18 will change to 18)
student_info['Int_School_Yr'] = student_info["School Yr"].str.split("-").str[1].astype('int64')
student_info['STUDENT-ZIP'] = student_info["STUDENT-ZIP"].astype('str')
student_info['STUDENT-ZIP'] = student_info["STUDENT-ZIP"].str.split("-").str[0].astype('str')
student_info['STUDENT-ZIP'] = student_info["STUDENT-ZIP"].str.split(" ").str[0]

student_info.loc[3270, 'STUDENT-ZIP'] = str(60619) 
student_info.loc[7091, 'STUDENT-ZIP'] = str(60608) 
student_info.loc[10254, 'STUDENT-ZIP'] = np.nan
student_info.loc[3627, 'STUDENT-ZIP'] = str(60619)
student_info.loc[10254, 'STUDENT-ZIP'] = str(60609)
student_info.loc[9486, 'STUDENT-ETHNIC'] = str(1)
student_info.loc[11933, 'STUDENT-ETHNIC'] = str(1)

student_info.loc[700, 'STUDENT-ETHNIC'] = str(1)
student_info.loc[3257, 'STUDENT-ETHNIC'] = str(4)
student_info.loc[3509, 'STUDENT-ETHNIC'] = str(1)
student_info.loc[4556, 'STUDENT-ETHNIC'] = str(4)
student_info.loc[11792, 'STUDENT-ETHNIC'] = str(1)
test = student_info.loc[dataset['STUDENT-ZIP'] == '60680 ', 'STUDENT-ZIP']

student_info.loc[dataset['STUDENT-ZIP'] == '60680 ', 'STUDENT-ZIP'] = '60608'
student_info.loc[dataset['STUDENT-ZIP'] == '60690 ', 'STUDENT-ZIP'] = '60609'
student_info.loc[dataset['STUDENT-ZIP'] == '60147 ', 'STUDENT-ZIP'] = '60148'
student_info.loc[dataset['STUDENT-ZIP'] == '60127 ', 'STUDENT-ZIP'] = '60126'
student_info.loc[dataset['STUDENT-ZIP'] == '60147 ', 'STUDENT-ZIP'] = '60148'
student_info.loc[dataset['STUDENT-ZIP'] == '60149 ', 'STUDENT-ZIP'] = '60148'

check = student_info['STUDENT-ZIP'].value_counts()

# Create an Student Title Column which will be 1, 2, 3, or 4  
# Where 1 = Fr, 2 = Sp, 3 = Jr, and 4 = Sr
student_info['Student_Title'] = 4 - (student_info['STUDENT-YR-GRAD'] - student_info['Int_School_Yr'])
student_info.drop(student_info.columns[0], axis = 1, inplace = True)

# Adding GPA Column which is Quality Pts / Credit Attempted
student_info['Yearly_GPA'] = np.round(student_info['STUDENT-QUAL-PTS'] / student_info['STUDENT-CRED-ATT'], 2)

# Empty Data replaced with NaN
# Empty data actually has ' ' and is not actually empty
student_info.replace(' ', np.NaN, inplace = True)

# Creating a boolean column where True = Student left school
student_info.loc[student_info['STUDENT-WITHDRAW-CODE'] == '  ', 'STUDENT-WITHDRAW-CODE'] = np.nan
student_info['Student_Withdrew'] = student_info['STUDENT-WITHDRAW-CODE'].notnull()

# Drop all features used to make aggregated columns
student_info.drop(['STUDENT-STATUS', 'STUDENT-WITHDRAW-CODE', 
                   'STUDENT-QUAL-PTS', 'STUDENT-CRED-ATT'], 
                    axis = 1, inplace = True)

#student_info.drop(['STUDENT-STATUS','STUDENT-CRED-ATT'], 
#                    axis = 1, inplace = True)

# Changing Nan Values for Count into 0's since it means they had no detentions
student_info['Count'] = student_info['Count'].fillna(0)
student_info.rename(columns={'Count':'Discipline'}, inplace=True)

# Fixing some NaN values
student_info[['STUDENT-PARISH','STUDENT-DISABILITY', 'STUDENT-RELIGION']] = student_info[['STUDENT-PARISH','STUDENT-DISABILITY', 'STUDENT-RELIGION']].fillna(value = 'None')
student_info[['STUDENT-PROGRAM']] = student_info[['STUDENT-PROGRAM']].fillna(value = 'CP')
student_info[['Yearly_GPA']] = student_info[['Yearly_GPA']].fillna(value = 0)

student_info[['STUDENT-ETHNIC']] = student_info[['STUDENT-ETHNIC']].fillna(value = 10)
student_info = student_info.astype({'STUDENT-ETHNIC': int})
student_info['STUDENT-ETHNIC'].fillna(5, inplace = True)
student_info['STUDENT-ETHNIC'].unique()
student_info.loc[student_info['STUDENT-ETHNIC'] > 4, 'STUDENT-ETHNIC'] = 5
student_info = student_info.astype({'STUDENT-ETHNIC': object})

columns = {'STUDENT-ID': 'id', 'STUDENT-ZIP':'zipcode', 'STUDENT-SEX':'sex',
                             'STUDENT-ELEM-SCHL':'elem_code', 'STUDENT-PARISH':'parish_code', 'STUDENT-YR-GRAD':'graduation_yr',
                             'STUDENT-DISABILITY':'disability', 'STUDENT-ETHNIC': 'ethnicity', 'STUDENT-PROGRAM':'program',
                             'STUDENT-RELIGION':'religion', 'STUDENT-PARENT-MARITAL-STATUS':'parent_status', 'Discipline':'discipline',
                             'Int_School_Yr':'current_yr', 'Student_Title':'grade_yr', 'Yearly_GPA':'gpa', 'Student_Withdrew':'target'}

student_info = student_info[['STUDENT-ID', 'STUDENT-ZIP', 'STUDENT-SEX',
                             'STUDENT-ELEM-SCHL', 'STUDENT-PARISH', 'STUDENT-YR-GRAD',
                             'STUDENT-DISABILITY', 'STUDENT-ETHNIC', 'STUDENT-PROGRAM',
                             'STUDENT-RELIGION', 'STUDENT-PARENT-MARITAL-STATUS', 'Discipline',
                             'Int_School_Yr', 'Student_Title', 'Yearly_GPA', 'Student_Withdrew']]

student_info = student_info.rename( columns = columns)

# Zip Codes
from uszipcode import SearchEngine
def to_lng(zipcode):
    search = SearchEngine(simple_zipcode=True)
    lng = search.by_zipcode(zipcode).to_dict()['lng']
    return lng
def to_lat(zipcode):
    search = SearchEngine(simple_zipcode=True)
    lat = search.by_zipcode(zipcode).to_dict()['lat']
    return lat

# Zipcodes to Distance
#final_set = student_info
student_info['lng'] = student_info['zipcode'].apply(to_lng)
student_info['lat'] = student_info['zipcode'].apply(to_lat)
student_info1 = student_info

def distance(latlng):
    d_lat = 41.831518
    d_lng = -87.623568
    return np.sqrt((d_lat - latlng[0])**2 + (d_lng - latlng[1])**2) *69

student_info1['distance'] = student_info1[['lat','lng']].apply(distance, axis = 1)
student_info1.drop(['lng', 'lat'], axis = 1, inplace = True)
student_info1 = student_info1[['id', 'zipcode', 'sex', 'elem_code', 'parish_code', 'disability',
                               'ethnicity', 'program', 'religion', 'parent_status', 'grade_yr','current_yr', 'gpa',
                               'discipline', 'distance', 'target']]
student_info1.to_csv('2019_Data_Numerical.csv')

# Saving Data
data_2008_2018 = student_info1
data_2008_2018.to_csv('2008-2018_Data_NumericalV9.csv', index = False)
data_2014_2018 = student_info1[student_info1.current_yr > 13 ]
data_2014_2018.to_csv('2014-2018_Data_NumericalV9.csv', index = False)



#######################################
#                                     #
#        Dropping Duplicates          #
#                                     #
#######################################

#new = dataset.groupby('STUDENT-ID').agg({'Discipline':'sum', 'Yearly_GPA': 'mean'}).reset_index()
new = data_2008_2018.groupby('id').agg({'discipline':'mean'}).reset_index()
data_2008_2018.drop_duplicates('id',keep = 'first', inplace = True)
data_2008_2018.drop(['discipline'], axis = 1, inplace = True)
data_2008_2018 = data_2008_2018.merge(new, on = 'id')
data_2008_2018 = data_2008_2018[['id', 'zipcode', 'sex', 'elem_code', 'parish_code', 'disability',
                               'ethnicity', 'program', 'religion', 'parent_status', 'grade_yr','current_yr', 'gpa',
                               'discipline', 'distance', 'target']] 
data_2008_2018.to_csv('2008-2018_Data_Unique_NumericalV9.csv', index = False)
##############
new = data_2014_2018.groupby('id').agg({'discipline':'mean'}).reset_index()
data_2014_2018.drop_duplicates('id',keep = 'first', inplace = True)
data_2014_2018.drop(['discipline'], axis = 1, inplace = True)
data_2014_2018 = data_2014_2018.merge(new, on = 'id')
data_2014_2018 = data_2014_2018[['id', 'zipcode', 'sex', 'elem_code', 'parish_code', 'disability',
                               'ethnicity', 'program', 'religion', 'parent_status', 'grade_yr','current_yr', 'gpa',
                               'discipline', 'distance', 'target']] 
data_2014_2018.to_csv('2014-2018_Data_Unique_NumericalV9.csv', index = False)

#######################################
#                                     #
#          Discretizing               #        
#                                     #
#######################################
def grade(gpa):
    float(gpa)
    if gpa >= 4.5:
        return 'A+'
    elif gpa < 4.5 and gpa >= 4.0:
        return 'A'
    elif gpa < 4.0 and gpa >= 3.5:
        return 'B+'
    elif gpa < 3.5 and gpa >= 3.0:
        return 'B'
    elif gpa < 3.0 and gpa >= 2.5:
        return 'C+'
    elif gpa < 2.5 and gpa >= 2.0:
        return 'C'
    elif gpa < 2.0 and gpa >= 1.5:
        return 'D+'
    elif gpa < 1.5 and gpa >= 1.0:
        return 'D'
    else:
        return 'F'

def disp_cat(count):
    int(count)
    if count < 1:
        return 'Level 0'
    elif count < 5:
        return 'Level 1'
    elif count < 10:
        return 'Level 2'
    elif count < 15:
        return 'Level 3'
    elif count < 20:
        return 'Level 4'
    elif count < 25:
        return 'Level 5'
    elif count < 30:
        return 'Level 6'
    else:
        return 'Level 7'

def dist_cat(count):
    if count < 2.0:
        return 'Less Than 2'
    elif count < 3.0:
        return 'Less than 3'
    elif count < 5.0:
        return 'Less than 5'
    elif count < 7.0:
        return 'Less than 7'
    elif count < 9.0:
        return 'Less than 9'
    elif count < 11.0:
        return 'Less than 11'
    else:
        return 'Greater than 11'
    
data_2008_2018['gpa'] = data_2008_2018['gpa'].apply(grade)
data_2008_2018['discipline'] = data_2008_2018['discipline'].apply(disp_cat)
data_2008_2018['distance'] = data_2008_2018['distance'].apply(dist_cat)
data_2008_2018.to_csv('2008-2018_Data_Unique_DiscreteV9.csv.csv', index = False)

data_2014_2018['gpa'] = data_2014_2018['gpa'].apply(grade)
data_2014_2018['discipline'] = data_2014_2018['discipline'].apply(disp_cat)
data_2014_2018['distance'] = data_2014_2018['distance'].apply(dist_cat)

student_info1['gpa'] = student_info1['gpa'].apply(grade)
student_info1['discipline'] = student_info1['discipline'].apply(disp_cat)
student_info1['distance'] = student_info1['distance'].apply(dist_cat)
student_info1.to_csv('2019_Data_Discrete.csv')



