import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("/Users/wangsfamily/Downloads/Tech Project/bank_transactions.csv", delimiter=',')

#("/Users/wangsfamily/Downloads/Tech Project/bank_transactions.csv", delimiter='\t'
# Print column names to verify
print("Column Names:")
print(data.columns)

# Clean the data (assuming it's already cleaned)
# Drop irrelevant columns for clustering
data.drop(['TransactionID', 'CustomerID', 'CustomerDOB', 'TransactionDate', 'TransactionTime'], axis=1, inplace=True)

# Perform feature preprocessing
# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['CustGender', 'CustLocation'])

# Normalize numerical variables
scaler = StandardScaler()
data[['CustAccountBalance', 'TransactionAmount (INR)']] = scaler.fit_transform(data[['CustAccountBalance', 'TransactionAmount (INR)']])

# Train K-means clustering model
kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming we want 5 clusters
kmeans.fit(data)

# Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Conduct cluster analysis to extract insights
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Scale back the cluster centers
cluster_analysis = pd.DataFrame(cluster_centers, columns=data.columns[:-1])  # Exclude the 'Cluster' column


print("Cluster Analysis:")
print(cluster_analysis) 

# Optionally, you can save the clustered dataset to a new CSV file
# data.to_csv("/Users/wangsfamily/Downloads/Tech Project/clustered_transaction_data.csv", index=False)
