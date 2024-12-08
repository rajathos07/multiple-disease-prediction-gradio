import os
import pickle
import gradio as gr

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved models
heart_model_path = os.path.join(working_dir, "saved_models", "heart_disease_model.sav")
parkinsons_model_path = os.path.join(working_dir, "saved_models", "parkinsons_model.sav")

heart_disease_model = pickle.load(open(heart_model_path, 'rb'))
parkinsons_model = pickle.load(open(parkinsons_model_path, 'rb'))

# Function for Heart Disease Prediction
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    try:
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        prediction = heart_disease_model.predict([user_input])
        result = "The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease"
        return result
    except Exception as e:
        return f"Error in prediction: {e}"

# Function for Parkinson's Disease Prediction
def predict_parkinsons(fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe):
    try:
        user_input = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer,
                      shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1,
                      spread2, d2, ppe]
        user_input = [float(x) for x in user_input]
        prediction = parkinsons_model.predict([user_input])
        result = "The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease"
        return result
    except Exception as e:
        return f"Error in prediction: {e}"

# Define Gradio interfaces
heart_interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(choices=[0, 1], label="Sex (0: Female, 1: Male)"),
        gr.Number(label="Chest Pain Type (0-3)"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Serum Cholesterol in mg/dl"),
        gr.Radio(choices=[0, 1], label="Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)"),
        gr.Number(label="Resting ECG Results (0-2)"),
        gr.Number(label="Max Heart Rate Achieved"),
        gr.Radio(choices=[0, 1], label="Exercise Induced Angina (0: No, 1: Yes)"),
        gr.Number(label="ST Depression Induced by Exercise"),
        gr.Number(label="Slope of the Peak Exercise ST Segment (0-2)"),
        gr.Number(label="Number of Major Vessels (0-4)"),
        gr.Number(label="Thal (0=Normal; 1=Fixed Defect; 2=Reversible Defect)")
    ],
    outputs="text",
    title="Heart Disease Prediction",
    description="Predict if a person has heart disease based on input features."
)

parkinsons_interface = gr.Interface(
    fn=predict_parkinsons,
    inputs=[
        gr.Number(label='MDVP:Fo(Hz)'),
        gr.Number(label='MDVP:Fhi(Hz)'),
        gr.Number(label='MDVP:Flo(Hz)'),
        gr.Number(label='MDVP:Jitter(%)'),
        gr.Number(label='MDVP:Jitter(Abs)'),
        gr.Number(label='MDVP:RAP'),
        gr.Number(label='MDVP:PPQ'),
        gr.Number(label='Jitter:DDP'),
        gr.Number(label='MDVP:Shimmer'),
        gr.Number(label='MDVP:Shimmer(dB)'),
        gr.Number(label='Shimmer:APQ3'),
        gr.Number(label='Shimmer:APQ5'),
        gr.Number(label='MDVP:APQ'),
        gr.Number(label='Shimmer:DDA'),
        gr.Number(label='NHR'),
        gr.Number(label='HNR'),
        gr.Number(label='RPDE'),
        gr.Number(label='DFA'),
        gr.Number(label='Spread1'),
        gr.Number(label='Spread2'),
        gr.Number(label='D2'),
        gr.Number(label='PPE')
    ],
    outputs="text",
    title="Parkinson's Disease Prediction",
    description="Predict if a person has Parkinson's disease based on input features."
)

# Launch Gradio app with both interfaces in a tabbed layout
app = gr.TabbedInterface([heart_interface, parkinsons_interface], ["Heart Disease", "Parkinson's Disease"])

# Run the app
if __name__ == "__main__":
    app.launch()
