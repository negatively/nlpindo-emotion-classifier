# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

pipe_lr = pickle.load(open("model/model_3.pkl", "rb"), encoding="bytes")
emotion = ["Neutral","Joy","Sadness","Fear","Surprise","Anger","Shame","Disgust"]
emoji_list = ['😶','😄','😥','😱','😲','😠','🙄','😖']


def predict(text_data):
    result = pipe_lr.predict([text_data])

    return result[0]

def get_predict_proba(text_data):
    result = pipe_lr.predict_proba([text_data])
    return result

def main():
    st.title("Classifier of Emotion in Text")
    menu = ["Home", "Monitor", "About"]
    option = st.sidebar.selectbox("Menu", menu)

    if option == "Home":
        with st.form(key='emosi_form'):
            text_raw  = st.text_area("Type your text below")
            text_submit = st.form_submit_button(label="Classify")

        preds = predict(text_raw)
        probability = get_predict_proba(text_raw)


        if text_submit:
            col_1,col_2 = st.beta_columns(2)
            with col_1:
                st.success("Original Text")
                st.write(text_raw)

                st.success("Prediction")
                st.write(f"{emotion[preds]} {emoji_list[preds]}")

                st.write(f"Confidence:{np.max(probability)}")

            with col_2:
                st.success("Prediction Probability")
                
                df_proba = pd.DataFrame(probability, columns=emotion)
                df_proba = df_proba.T.reset_index()
                df_proba.columns = ["emotion","probability"]
                
                fig = alt.Chart(df_proba).mark_bar().encode(x='emotion', y='probability',
                                color='emotion')
                st.altair_chart(fig, use_container_width=True)


if __name__=="__main__":
    main()