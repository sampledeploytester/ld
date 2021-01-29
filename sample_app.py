import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from joblib import load

# Explanation
st.title('L-detector')
st.write("""
    ### Description
    This web app accepts one or more respondent data (.txt) and return predictions.
""")

# File upload, this will be a list of objects
uploaded_file = st.file_uploader('Upload respondent file(s)', accept_multiple_files=True, type='txt')

ls_of_bytes = []
for i in uploaded_file:
    # an item in the list can be turned into bytesio
    # using .read()
    bytes_data = i.read()
    ls_of_bytes.append(bytes_data)

def feature_extractor(uploaded_raw_data):
    X = []
    respondent_ids = []
    for respondent_data in uploaded_raw_data:
        df = pd.read_csv(BytesIO(respondent_data), sep='\t')
        respondent_ids.append(np.unique(df['respondentId'])[0])
        df['compatibility'].replace(to_replace=['na', 'comp', 'incomp'],
                                        value=[0, 1, 2],
                                        inplace=True)

        # Extract features from txt
        feats = []
        for block_no in np.unique(df.iloc[:, 5]):
            feats.append(block_no)
            feats.append(df['compatibility'][df['blockNumber'] == block_no].iloc[0])
            feats.append(np.mean(df['rt1'][df['blockNumber'] == block_no].values))
            feats.append(np.std(df['rt1'][df['blockNumber'] == block_no].values))
            feats.extend(df['rt1'][df['blockNumber'] == block_no].values)
            feats.extend(df['rt2'][df['blockNumber'] == block_no].values)
            feats.extend(df['firstError'][df['blockNumber'] == block_no].values)
        X.append(feats)
    num_of_respondents = len(respondent_ids)
    X = np.array(X)
    X = X.reshape(num_of_respondents, -1)
    return X, respondent_ids

clf = load('ld_ver1.joblib')

def resulting_df(respondent_ids, l_prob, preds):

    df_col_1 = np.array(respondent_ids).reshape(-1, 1)

    df_col_2 = l_prob.reshape(-1, 1)

    preds_string = []
    for pred in preds:
        if pred == 1:
            preds_string.append('Group 1')
        elif pred == 0:
            preds_string.append('Group 2')
    df_col_3 = np.array(preds_string).reshape(-1, 1)
    
    arr = np.hstack((df_col_1, df_col_2, df_col_3))
    
    df = pd.DataFrame(arr,
                      columns = ['Respondent ID', 'L-prob.', 'Prediction'])
    return df

th = st.slider('Discrimination threshold (default = 0.42)', 0.01, 0.99, value=0.42)
btn_run = st.button('Classify')

if btn_run:
    try:
        X, respondent_ids = feature_extractor(ls_of_bytes)
        pred_probs = clf.predict_proba(X)[:, 1]
        preds = (pred_probs > th) * 1
        l_prob = np.round(pred_probs, 3)
        df = resulting_df(respondent_ids, l_prob, preds)
        st.write(df)
    except:
        st.write('Txt file is not properly formatted, please refresh and upload new files.')