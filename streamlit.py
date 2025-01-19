# from flask import Flask, render_template, request
# import pickle
# import pandas as pd

# app = Flask(__name__)

# # Load pickle files
# with open('grid_search_rf.pkl', 'rb') as file:
#     model = pickle.load(file)

# with open('ordinal_encoder.pkl', 'rb') as file:
#     ordinal_encoder = pickle.load(file)

# with open('onehot_encoder.pkl', 'rb') as file:
#     onehot_encoder = pickle.load(file)

# with open('freq_encoding.pkl', 'rb') as file:
#     freq_encoding = pickle.load(file)

# with open('minmax_scaler.pkl', 'rb') as file:
#     minmax_scaler = pickle.load(file)

# # Dropdown options
# gender_options = ['Male', 'Female', 'Others']
# education_options = ordinal_encoder.categories_[0]
# job_title_options = ordinal_encoder.categories_[1]
# department_options = list(freq_encoding.keys())

# @app.route('/')
# def home():
#     return render_template(
#         'index.html',
#         gender_options=gender_options,
#         education_options=education_options,
#         job_title_options=job_title_options,
#         department_options=department_options
#     )

# @app.route('/result', methods=['POST'])
# def result():
#     try:
#         # Collect user inputs
#         input_data = {
#             'Gender': request.form.get('Gender', '').strip(),
#             'Education_Level': request.form.get('Education_Level', '').strip(),
#             'Job_Title': request.form.get('Job_Title', '').strip(),
#             'Department': request.form.get('Department', '').strip(),
#             'Age': float(request.form.get('Age', 0)),
#             'Years_At_Company': float(request.form.get('Years_At_Company', 0)),
#             'Performance_Score': float(request.form.get('Performance_Score', 0)),
#             'Monthly_Salary': float(request.form.get('Monthly_Salary', 0)),
#             'Work_Hours_Per_Week': float(request.form.get('Work_Hours_Per_Week', 0)),
#             'Projects_Handled': float(request.form.get('Projects_Handled', 0)),
#             'Overtime_Hours': float(request.form.get('Overtime_Hours', 0)),
#             'Sick_Days': float(request.form.get('Sick_Days', 0)),
#             'Remote_Work_Frequency': float(request.form.get('Remote_Work_Frequency', 0)),
#             'Team_Size': float(request.form.get('Team_Size', 0)),
#             'Training_Hours': float(request.form.get('Training_Hours', 0)),
#             'Promotions': float(request.form.get('Promotions', 0)),
#             'Employee_Satisfaction_Score': float(request.form.get('Employee_Satisfaction_Score', 0)),
#         }

#         # Check if required dropdown fields are empty
#         if not input_data['Gender'] or not input_data['Education_Level'] or not input_data['Job_Title'] or not input_data['Department']:
#             return "Error: Please ensure all dropdown fields are selected.", 400

#         # Encoding inputs
#         gender_onehot = onehot_encoder.transform([[input_data['Gender']]])
#         education_job_encoded = ordinal_encoder.transform(
#             [[input_data['Education_Level'], input_data['Job_Title']]]
#         )
#         department_freq = freq_encoding.get(input_data['Department'], 0)

#         # Combine all features into a single row
#         row = [
#             input_data['Age'],
#             education_job_encoded[0][1],  # Job_Title
#             input_data['Years_At_Company'],
#             education_job_encoded[0][0],  # Education_Level
#             input_data['Performance_Score'],
#             input_data['Monthly_Salary'],
#             input_data['Work_Hours_Per_Week'],
#             input_data['Projects_Handled'],
#             input_data['Overtime_Hours'],
#             input_data['Sick_Days'],
#             input_data['Remote_Work_Frequency'],
#             input_data['Team_Size'],
#             input_data['Training_Hours'],
#             input_data['Promotions'],
#             input_data['Employee_Satisfaction_Score'],
#             *gender_onehot[0],  # Gender one-hot encoded
#             department_freq  # Department frequency encoding
#         ]

#         # Scale numeric features
#         row_scaled = minmax_scaler.transform([row])

#         # Predict using the loaded model
#         prediction = model.predict(row_scaled)[0]

#         # Format the prediction result for better readability
#         prediction_text = "The employee is likely to resign." if prediction == 1 else "The employee is not likely to resign."

#         return render_template('result.html', prediction=prediction_text)

#     except ValueError as e:
#         return f"Error: Invalid input data. {str(e)}", 400

# if __name__ == '__main__':
#     app.run(debug=True)


import streamlit as st
import pickle
import pandas as pd

# Load pickle files
with open('grid_search_rf.pkl', 'rb') as file:
    model = pickle.load(file)

with open('ordinal_encoder.pkl', 'rb') as file:
    ordinal_encoder = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

with open('freq_encoding.pkl', 'rb') as file:
    freq_encoding = pickle.load(file)

with open('minmax_scaler.pkl', 'rb') as file:
    minmax_scaler = pickle.load(file)

# Dropdown options
gender_options = ['Male', 'Female', 'Others']
education_options = ordinal_encoder.categories_[0]
job_title_options = ordinal_encoder.categories_[1]
department_options = list(freq_encoding.keys())

# Streamlit UI
st.title("Employee Resignation Prediction")
st.write("This app predicts whether an employee is likely to resign based on input features.")

# Input fields
st.sidebar.header("Employee Details")

Gender = st.sidebar.selectbox("Gender", gender_options)
Education_Level = st.sidebar.selectbox("Education Level", education_options)
Job_Title = st.sidebar.selectbox("Job Title", job_title_options)
Department = st.sidebar.selectbox("Department", department_options)

Age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=30, step=1)
Years_At_Company = st.sidebar.number_input("Years at Company", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
Performance_Score = st.sidebar.number_input("Performance Score (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
Monthly_Salary = st.sidebar.number_input("Monthly Salary", min_value=0.0, value=50000.0, step=1000.0)
Work_Hours_Per_Week = st.sidebar.number_input("Work Hours Per Week", min_value=0.0, max_value=168.0, value=40.0, step=1.0)
Projects_Handled = st.sidebar.number_input("Projects Handled", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
Overtime_Hours = st.sidebar.number_input("Overtime Hours", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
Sick_Days = st.sidebar.number_input("Sick Days", min_value=0.0, max_value=365.0, value=5.0, step=1.0)
Remote_Work_Frequency = st.sidebar.number_input("Remote Work Frequency (Days Per Week)", min_value=0.0, max_value=7.0, value=2.0, step=1.0)
Team_Size = st.sidebar.number_input("Team Size", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
Training_Hours = st.sidebar.number_input("Training Hours", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
Promotions = st.sidebar.number_input("Promotions", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
Employee_Satisfaction_Score = st.sidebar.number_input("Employee Satisfaction Score (0-10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)

# Prediction button
if st.button("Predict"):
    try:
        # Encode categorical inputs
        gender_onehot = onehot_encoder.transform([[Gender]])
        education_job_encoded = ordinal_encoder.transform([[Education_Level, Job_Title]])
        department_freq = freq_encoding.get(Department, 0)

        # Combine all inputs into a single feature row
        row = [
            Age,
            education_job_encoded[0][1],  # Job_Title
            Years_At_Company,
            education_job_encoded[0][0],  # Education_Level
            Performance_Score,
            Monthly_Salary,
            Work_Hours_Per_Week,
            Projects_Handled,
            Overtime_Hours,
            Sick_Days,
            Remote_Work_Frequency,
            Team_Size,
            Training_Hours,
            Promotions,
            Employee_Satisfaction_Score,
            *gender_onehot[0],  # Gender one-hot encoded
            department_freq  # Department frequency encoding
        ]

        # Scale numeric features
        row_scaled = minmax_scaler.transform([row])

        # Predict using the loaded model
        prediction = model.predict(row_scaled)[0]

        # Display the result
        if prediction == 1:
            st.error("The employee is likely to resign.")
        else:
            st.success("The employee is not likely to resign.")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
