import pandas as pd
import numpy as np
import math
import streamlit as st


class ID3Classifier:
    def __init__(self):
        self.tree = None

    def calculate_entropy(self, col):
        values, counts = np.unique(col, return_counts=True)
        ent = 0
        for count in counts:
            p = count / len(col)
            ent -= p * math.log2(p)
        return ent

    def information_gain(self, df, attribute, target):
        total_entropy = self.calculate_entropy(df[target])
        values, counts = np.unique(df[attribute], return_counts=True)
        weighted_entropy = 0
        for i in range(len(values)):
            subset = df[df[attribute] == values[i]]
            weighted_entropy += (counts[i]/len(df)) * self.calculate_entropy(subset[target])
        return total_entropy - weighted_entropy

    def fit(self, df, target, attributes):
        self.tree = self._build_tree(df, target, attributes)

    def _build_tree(self, df, target, attributes):
        if len(np.unique(df[target])) == 1:
            return df[target].iloc[0]
        if len(attributes) == 0:
            return df[target].mode()[0]

        gains = [self.information_gain(df, attr, target) for attr in attributes]
        best_attr = attributes[np.argmax(gains)]
        tree = {best_attr: {}}

        for value in np.unique(df[best_attr]):
            subset = df[df[best_attr] == value]
            remaining_attrs = [attr for attr in attributes if attr != best_attr]
            tree[best_attr][value] = self._build_tree(subset, target, remaining_attrs)
        return tree

    def predict(self, tree, sample):
        if not isinstance(tree, dict):
            return tree
        attr = next(iter(tree))
        value = sample.get(attr)
        if value in tree[attr]:
            return self.predict(tree[attr][value], sample)
        else:
            return "Unknown"


def main():
    st.set_page_config(page_title="ID3 Weather Predictor", page_icon="🎾")
    st.title("🎾 ID3 Decision Tree: Play Tennis?")
    
    # 1. Dataset Section
    st.subheader("Training Data")
    data = pd.DataFrame({
        'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain',
                    'Overcast','Sunny','Sunny','Rain','Sunny','Overcast',
                    'Overcast','Rain'],
        'Humidity': ['High','High','High','High','Normal','Normal',
                     'Normal','High','Normal','High','Normal','High',
                     'Normal','High'],
        'PlayTennis': ['No','No','Yes','Yes','Yes','No',
                        'Yes','No','Yes','Yes','Yes','Yes',
                        'Yes','No']
    })
    st.dataframe(data, use_container_width=True)

    
    clf = ID3Classifier()
    features = ['Outlook', 'Humidity']
    clf.fit(data, 'PlayTennis', features)

    st.subheader("Generated Logic (JSON Tree)")
    st.json(clf.tree)
    
    

    
    st.divider()
    st.subheader("Make a New Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        ui_outlook = st.selectbox("Current Outlook", options=['Sunny', 'Overcast', 'Rain'])
    with col2:
        ui_humidity = st.selectbox("Current Humidity", options=['High', 'Normal'])

    sample = {'Outlook': ui_outlook, 'Humidity': ui_humidity}
    result = clf.predict(clf.tree, sample)

    
    if result == "Yes":
        st.balloons()
        st.success(f"### Prediction: You should play tennis! ✅")
    elif result == "No":
        st.error(f"### Prediction: Don't play tennis today. ❌")
    else:
        st.warning(f"### Prediction: {result}")

if __name__ == "__main__":
    main()
