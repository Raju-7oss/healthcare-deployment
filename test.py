import joblib 
import numpy as np
 #load the saved model

model=joblib.load("model_scaled.pkl")
scaler=joblib.load("scaled.pkl")
user_input=np.array([[1,2,3000,499,0.0075,6,7,8]])
scaled_input=scaler.transform(user_input)
#prediction
result= model.predict(scaled_input)

print("The result is" ,result[0])