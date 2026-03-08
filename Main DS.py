import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\gaura\OneDrive\Documents\SECOND SEM\FUN FOR DATA SCIENCE\newdata.csv")
#print(df.head())
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
   df[col].fillna(df[col].median())
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
   df[col].fillna(df[col].mode()[0])
df['Year'] = pd.to_datetime(df['Date'], dayfirst=True).dt.year
#print(df.isnull().sum())
le=LabelEncoder()
encoder=OrdinalEncoder(categories=[['Top Order','Middle Order','Lower-Middle Order',"Tail-Ender"]])
df['Role']=encoder.fit_transform(df[['Role']])
df['Role'] = df['Role'].astype(int)
df['Is_Spin']=df['Bowler_Type'].str.contains('Spin',case=False).astype(int)
df['Is_Left']=df['Bowler_Type'].str.contains('Left',case=False).astype(int)
encoders={}
for col in['Opponent','Venue','Pitch_Type','Toss_Decision']:
   le=LabelEncoder()
   df[col+'_enc']=le.fit_transform(df[col].astype(str))
   encoders[col]=le

y=df['Runs']
features=[
   'Year',
   'Pitch_Rating',
   'Innings',
   'Role',
   'Toss_Decision_enc',
   'Opponent_enc',
   'Venue_enc',
   'Pitch_Type_enc',
   'Is_Spin',
   'Is_Left'
]
X= df[features]
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
rf_model=RandomForestRegressor(n_estimators=100,max_depth=4, random_state=42)
rf_model.fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
#print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")
print(f"Average Error: {mean_absolute_error(y_test, y_pred):.2f} Runs")
print(f"Training R2: {rf_model.score(X_train, y_train):.2f}")
print(f"Testing R2: {rf_model.score(X_test, y_test):.2f}")
#print(df.head())

#----------------------GRAPH----------------------


#----------LINE GRAPH BATSMEN----------#

# imp=pd.Series(rf_model.feature_importances_,index=features).sort_values(ascending=False)
# plt.figure(figsize=(10,6))
# imp.plot(kind='bar', color='red')
# plt.title('Analytical Reasons for the Batting Downfall')
# plt.ylabel('Importance Score')
# #plt.legend()
# plt.show()

# #----------DECISION TREE----------#
# sub_tree = rf_model.estimators_[0]
# plt.figure(figsize=(20,10))
# plot_tree(sub_tree, feature_names=features, max_depth=2, filled=True)
# plt.show()

# #----------BAR GRAPH---------------#
# plt.figure(figsize=(10,6))
# df.groupby('Year')['Runs'].mean().plot(kind='line', marker='o', color='red', linewidth=2)
# plt.title('Average runs by Indian Batsmen')
# plt.grid(True)
# plt.show()

# #-----------HEATMAP-----------------#
# plt.figure(figsize=(10,8))
# # Select only numerical columns for correlation
# corr_matrix = df.select_dtypes(include=['int32', 'int64', 'float64']).corr()
# sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f")
# plt.title('Correlation Heatmap: What factors are linked to Runs?')
# plt.show()

# #-----------BAR GRAPH PITCH-----------#
# plt.figure(figsize=(8, 5))
# sns.barplot(x='Pitch_Rating', y='Runs', data=df, palette='RdYlGn',errorbar=None) # Built-in Red-Yellow-Green palette
# plt.title("Impact of Pitch Quality on Average Runs")
# plt.xlabel("Pitch Rating (1=Hard, 5=Easy)")
# plt.ylabel("Average Runs")
# plt.tight_layout()
# plt.show()