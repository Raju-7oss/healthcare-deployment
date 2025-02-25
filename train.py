import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# loading the dataset from url
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
df=pd.read_csv(url,names=names)

x=df.iloc[:,0:8]
y=df.iloc[:,8]

#split the data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#train the model
model=LogisticRegression()
model.fit(x_train_scaled,y_train)

print(f"[INFO] model training is completed")
result=model.score(x_test_scaled,y_test)
print(f"[INFO] test score is {result}")
# serialization
import joblib
joblib.dump(model,"model_scaled.pkl") #pickle file 
joblib.dump(scaler,"scaled.pkl")