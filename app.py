import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the Streamlit app
def main():
    st.title('Tasks Output')

    uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)

        # Task 1
        st.header('Task 1 Output')
        st.write("Data from uploaded file:")
        st.write(data.head())

        features = data.iloc[:, :-1]

        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(features)

        plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KMeans Clustering')
        st.pyplot()

        # Task 2
        st.header('Task 2 Output')

        train_data = pd.read_excel("train.xlsx")
        X_train = train_data.iloc[:, :-1]
        y_train = train_data['target']

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        train_preds = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        st.write(f"Train Accuracy: {train_accuracy}")

        # Task 3
        st.header('Task 3 Output')

        # Your Task 3 code here

if __name__ == '__main__':
    main()
