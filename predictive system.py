import numpy as np
import pickle
#loading the saved model
loaded_model = pickle.load(open("C:\\Users\\SHREYASI DAS\\OneDrive\\Documents\\Machine Learning\\Diabetes Prediction Model\\trained_model.sav" ,'rb'))

input_data=(5,166,72,19,175,25.8,0.587,51)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array to predict for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction =loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print("The person is not Diabetic")
else:
  print(" The person is Diabetic")  