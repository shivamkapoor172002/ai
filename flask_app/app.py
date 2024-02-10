from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    data = pd.read_excel("test.xlsx")
    features = data.iloc[:, :-1]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KMeans Clustering')
    plot_path = 'static/task1_plot.png'
    plt.savefig(plot_path)
    plt.close()

    train_data = pd.read_excel("train.xlsx")
    X_train = train_data.iloc[:, :-1]
    y_train = train_data['target']
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_preds)

    raw_data = pd.read_excel("rawdata.xlsx")
    raw_data['date'] = raw_data['date'].dt.strftime('%Y-%m-%d')
    raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'].astype(str))
    duration_per_location = raw_data.groupby(['date', 'location'])['datetime'].agg(lambda x: (x.max() - x.min()).seconds / 60)
    activity_counts = raw_data.groupby(['date', 'activity']).size().unstack(fill_value=0)
    output_data = pd.DataFrame(index=raw_data['date'].unique())
    output_data.index.name = 'date'
    output_data['pick_activities'] = activity_counts['picked']
    output_data['place_activities'] = activity_counts['placed']

    if 'inside' not in duration_per_location.index.levels[1]:
        output_data['inside_duration'] = float('nan')
    if 'outside' not in duration_per_location.index.levels[1]:
        output_data['outside_duration'] = float('nan')
    else:
        inside_duration = duration_per_location.xs('inside', level='location', drop_level=False).groupby('date').sum()
        outside_duration = duration_per_location.xs('outside', level='location', drop_level=False).groupby('date').sum()
        output_data['inside_duration'] = inside_duration
        output_data['outside_duration'] = outside_duration

    output_data = output_data.fillna(0)
    output_path = 'static/output.xlsx'
    output_data.to_excel(output_path)

    return render_template('index.html', plot_path=plot_path, train_accuracy=train_accuracy, output_path=output_path)

if __name__ == '__main__':
    app.run(debug=True)
