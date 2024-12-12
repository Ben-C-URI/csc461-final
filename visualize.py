import streamlit as st
import pandas as pd
import openai
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

sct_data = pd.read_csv('data/df.csv')
sct_data['hash_key'] = sct_data['ccode'].astype(str) + sct_data['year'].astype(str)
sct_data['ciri_cat'] = pd.cut(sct_data['ciri'], bins=4, labels=['very low', 'low', 'high', 'very high'])
data = sct_data.copy()

# clean and preprocess
data['bankingcrisis'] = data['bankingcrisis'].fillna(0)
data['systemiccrisis'] = data['systemiccrisis'].fillna(0)
data = data.drop(['goldstandard', 'domestic_debt_in_default', 'independence'], axis=1)
data = data.dropna()

X = data[['ciri_cat', 'inflation', 'gdpgrowth', 'log_itpop', 'region', 'imilex', 'coup', 'coup_lag', 
          'bankingcrisis', 'systemiccrisis', 'gwf_autocracy', 'gwf_military', 'gwf_monarchy', 
          'gwf_party', 'gwf_personal', 'gwf_democracy']]
y = data['ongoing_sanction']

label_encoder = LabelEncoder()
data['ciri_cat_encoded'] = label_encoder.fit_transform(data['ciri_cat'])
X['ciri_cat'] = data['ciri_cat_encoded']
X = pd.get_dummies(X, columns=['region'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# train dt
tree1 = DecisionTreeClassifier(max_depth=10, random_state=400)
tree1.fit(X_train, y_train)

# check replacement of hugface with gpt
def fetch_historical_context_gpt(country_name, year, features, sanction_status):
    openai.api_key = "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"

    prompt = f"""
    Provide an explanation for the decision tree's prediction:
    - Country: {country_name}
    - Year: {year}
    - Key Features: {features}
    - Prediction: {sanction_status}

    You understand ciri_cat represents lower civil rights (1) to higher civil rights (4), and understand log_itpop is log population. (imilex) is military expenditure. Any variable starting with (gwf_) refers to the GWF dataset and is a binary depending on what type of regime structure is in place.
    Heavily rely on the results of the decision tree model to attempt to make an informed prediction.
    Explain the most influential features contributing to the decision. If sanctions are predicted, provide historical context relevant to the country and year. Structure the explanation with clear headings: Prediction Summary, Key Features, Historical Context. Please attempt to provide sources when available to you!
    After using the decision tree, then reference the actual outcome of the prediction, and speculate shortly as to some of the political events that could have led the decision tree to be subverted.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that explains decision tree predictions in a structured and concise way, while also maintaining political conventions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"An error occurred: {e}"

# streamlit! (if there is time) =================================================================
st.title("Prototype: Ongoing Sanction Prediction and Explanation")

year = st.number_input("Enter year:", min_value=1981, max_value=2020)

valid_data = sct_data[sct_data['year'] == year]
if not valid_data.empty:
    ccode_mapping = valid_data[['ccode', 'gwf_country']].drop_duplicates()
    ccode_options = {row['ccode']: row['gwf_country'] for _, row in ccode_mapping.iterrows()}
    selected_ccode = st.selectbox("Select a country code (ccode):", options=list(ccode_options.keys()),
                                  format_func=lambda c: f"{c} - {ccode_options[c]}")
else:
    st.write("No data available for the selected year.")
    selected_ccode = None

if st.button("Generate Prediction") and selected_ccode:
    input_data = sct_data[(sct_data['ccode'] == selected_ccode) & (sct_data['year'] == year)].copy()
    if not input_data.empty:
        input_data['ciri_cat_encoded'] = label_encoder.transform(input_data['ciri_cat'])
        input_data['ciri_cat'] = input_data['ciri_cat_encoded']
        input_data = pd.get_dummies(input_data, columns=['region'], drop_first=True)
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X_train.columns]

        predicted_class = tree1.predict(input_data)
        sanction_status = "Sanction likely" if predicted_class[0] == 1 else "No sanction predicted"

        features_description = {k: v for k, v in input_data.to_dict(orient='records')[0].items() if v != 0}

        country_name = ccode_options[selected_ccode]
        explanation = fetch_historical_context_gpt(country_name, year, features_description, sanction_status)

        st.write(f"### Prediction Explanation for {country_name} in {year}")
        st.write(explanation)
    else:
        st.write("No data available for the selected country and year.")
