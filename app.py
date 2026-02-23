import pandas as pd
import numpy as np
import math

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
        # Case 1: All target values are the same
        if len(np.unique(df[target])) == 1:
            return df[target].iloc[0]

        # Case 2: No attributes left to split
        if len(attributes) == 0:
            return df[target].mode()[0]

        # Select best attribute based on Info Gain
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

if __name__ == "__main__":
    # --- Step 1: Dataset ---
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

    # --- Step 2: Training ---
    clf = ID3Classifier()
    features = list(data.columns)
    features.remove('PlayTennis')
    clf.fit(data, 'PlayTennis', features)

    print("Generated Decision Tree:")
    print(clf.tree)

    # --- Step 3: Testing ---
    sample = {'Outlook': 'Sunny', 'Humidity': 'High'}
    result = clf.predict(clf.tree, sample)
    print(f"\nSample: {sample}")
    print(f"Result (PlayTennis): {result}")
