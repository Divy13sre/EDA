step:Import Libraries and load dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())

output:

  PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  


step 1:Summary statistics


print(df.info())
print(df.describe())
print(df.isnull().sum())

output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
       PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  714.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
std     257.353842    0.486592    0.836071   14.526497    1.102743   
min       1.000000    0.000000    1.000000    0.420000    0.000000   
25%     223.500000    0.000000    2.000000   20.125000    0.000000   
50%     446.000000    0.000000    3.000000   28.000000    0.000000   
75%     668.500000    1.000000    3.000000   38.000000    1.000000   
max     891.000000    1.000000    3.000000   80.000000    8.000000   

            Parch        Fare  
count  891.000000  891.000000  
mean     0.381594   32.204208  
std      0.806057   49.693429  
min      0.000000    0.000000  
25%      0.000000    7.910400  
50%      0.000000   14.454200  
75%      0.000000   31.000000  
max      6.000000  512.329200  
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

step 3:Histograms and boxplots for numerical features

# Histograms
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Boxplots (example for age and fare)
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age and Fare")
plt.show()

output:
![image](https://github.com/user-attachments/assets/cfd9e686-cf88-4c2d-a3b4-3b8f5301e3ba)
![image](https://github.com/user-attachments/assets/a65c0b5a-ff38-40ec-a3b6-7aa5db62809f)
![image](https://github.com/user-attachments/assets/3102fb64-63f6-407c-afd7-f7534943920b)
![image](https://github.com/user-attachments/assets/63de67e4-526f-4606-a060-ce18f6701a1f)
![image](https://github.com/user-attachments/assets/462707d0-620b-4e88-982f-2178d0e9b73f)

step 3:Pairplot and correlation matrix

# Correlation matrix
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pairplot (filtering numeric columns or specific columns for clarity)
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']].dropna(), hue='Survived')
plt.show()

output:
![image](https://github.com/user-attachments/assets/2c3da487-cf54-4978-b4df-be653e236c3f)
![image](https://github.com/user-attachments/assets/4a371827-7453-45d6-a299-2198eb00d5c7)

step 4: Identifies patterns,trends or anomalies
.outliers in boxplots(very high fare values)
.skewness in age distribution
.class imbalance in survived
.strong correlations(eg:Pclass and fare)

step 5:Feature level insights

eg:.Passengers in 1st class had higher survival rates.

   .Females had higher survival probability than males.

   .Younger passengers had slightly higher survival chances.

   # Barplot survival rate by sex
sns.barplot(data=df, x='Sex', y='Survived')
plt.title("Survival Rate by Sex")
plt.show()

# Survival by Pclass
sns.barplot(data=df, x='Pclass', y='Survived')
plt.title("Survival Rate by Passenger Class")
plt.show()

output:
![image](https://github.com/user-attachments/assets/dd3a9959-984f-4053-ac8b-c0457356d151)
![image](https://github.com/user-attachments/assets/f4e5d961-0a40-4ee7-84bd-b1a767dce7b1)
















