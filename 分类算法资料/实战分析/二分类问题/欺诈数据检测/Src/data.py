import os
import pandas as pd
import numpy as np

# Create folders if they don't exist
folders = ['Folder1', 'Folder2', 'Folder3', 'Folder4', 'Folder5']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Generate Transaction Data
transaction_records = pd.DataFrame({
    'TransactionID': range(1, 1001),
    'Amount': np.random.uniform(10, 100, 1000),
    'CustomerID': np.random.randint(1001, 2001, 1000)
})

transaction_metadata = pd.DataFrame({
    'TransactionID': range(1, 1001),
    'Timestamp': pd.date_range('2022-01-01', periods=1000, freq='H'),
    'MerchantID': np.random.randint(2001, 3001, 1000)
})

# Generate Customer Profiles
customer_data = pd.DataFrame({
    'CustomerID': range(1001, 2001),
    'Name': ['Customer ' + str(i) for i in range(1001, 2001)],
    'Age': np.random.randint(18, 65, 1000),
    'Address': ['Address ' + str(i) for i in range(1001, 2001)]
})

account_activity = pd.DataFrame({
    'CustomerID': range(1001, 2001),
    'AccountBalance': np.random.uniform(1000, 10000, 1000),
    'LastLogin': pd.date_range('2022-01-01', periods=1000, freq='D')
})

# Generate Fraudulent Patterns
fraud_indicators = pd.DataFrame({
    'TransactionID': range(1, 1001),
    'FraudIndicator': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
})

suspicious_activity = pd.DataFrame({
    'CustomerID': range(1001, 2001),
    'SuspiciousFlag': np.random.choice([0, 1], 1000, p=[0.98, 0.02])
})

# Generate Transaction Amounts
amount_data = pd.DataFrame({
    'TransactionID': range(1, 1001),
    'TransactionAmount': np.random.uniform(10, 100, 1000)
})

anomaly_scores = pd.DataFrame({
    'TransactionID': range(1, 1001),
    'AnomalyScore': np.random.uniform(0, 1, 1000)
})

# Generate Merchant Information
merchant_data = pd.DataFrame({
    'MerchantID': range(2001, 3001),
    'MerchantName': ['Merchant ' + str(i) for i in range(2001, 3001)],
    'Location': ['Location ' + str(i) for i in range(2001, 3001)]
})

transaction_category_labels = pd.DataFrame({
    'TransactionID': range(1, 1001),
    'Category': np.random.choice(['Food', 'Retail', 'Travel', 'Online', 'Other'], 1000)
})

# Save the generated data into CSV files
transaction_records.to_csv('Folder1/transaction_records.csv', index=False)
transaction_metadata.to_csv('Folder1/transaction_metadata.csv', index=False)
customer_data.to_csv('Folder2/customer_data.csv', index=False)
account_activity.to_csv('Folder2/account_activity.csv', index=False)
fraud_indicators.to_csv('Folder3/fraud_indicators.csv', index=False)
suspicious_activity.to_csv('Folder3/suspicious_activity.csv', index=False)
amount_data.to_csv('Folder4/amount_data.csv', index=False)
anomaly_scores.to_csv('Folder4/anomaly_scores.csv', index=False)
merchant_data.to_csv('Folder5/merchant_data.csv', index=False)
transaction_category_labels.to_csv('Folder5/transaction_category_labels.csv', index=False)
