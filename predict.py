import streamlit as st
import pickle
import numpy as np
import pandas as pd


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

stacked = data["model"]
scaler = data["scaler"]
columns = ['critical_flag', 'score', 'community_board', 'inspection_year', 'inspection_month', 'inspection_day', 'grade']

def show_predict_page():
    st.title("NYC Closing Restaurent Prediction ")

    st.write("""### We need some information for prediction""")

    grade = (
        "A",
        "B",
         "C",
        "G",
         "N",
        "P",
        "Z",
        "UNGRADE"
    )

    flag = (
        "Critical",
        "Not Critical",
    )

    community_board = (
        101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
              110.0, 111.0, 112.0, 164.0, 201.0, 202.0, 203.0, 204.0, 205.0,
              206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 226.0, 227.0,
              228.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0,
              309.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0, 317.0,
              318.0, 355.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0,
              408.0, 409.0, 410.0, 411.0, 412.0, 413.0, 414.0, 480.0, 481.0,
              482.0, 483.0, 501.0, 502.0, 503.0, 595.0
    )

    grades = st.selectbox("Grade", grade)
    flags = st.selectbox("Flag", flag)
    community_boards = st.selectbox("Community Board", community_board)
    score = st.slider("Score", 0, 166)
    date = st.date_input('Date')

    ok = st.button("Predict")

    if flags == "Critical":
        flags = 1
    else:
        flags = 0

    if grades == "UNGRADE":
        grades = 1
    else:
        grades = 0

    if ok:
        X = np.array([[ flags, score, community_boards, date.year, date.month, date.day ,grades ]])
        X = pd.DataFrame(scaler.transform(X), columns=columns)

        prediction = stacked.predict(X)

        if prediction == 1:
            prediction = "Will Close"
        else:
            prediction = "It Will Not Be Closed"
            
        st.subheader(f"Is the restaurent going to close? \n {prediction}")