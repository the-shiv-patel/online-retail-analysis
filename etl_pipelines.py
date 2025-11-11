import mysql.connector
import pandas as pd
import numpy as np
from datetime import timedelta
import datetime
import math
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.cluster import KMeans
import joblib


def get_connection():
  return mysql.connector.connect(
      host="",
      user="",
      password="",
      database=""
  )

def getLastIndex():
    last_index_query = 'SELECT MAX(index_no) AS last_index FROM raw_data;'
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute(last_index_query)
    result_dict = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if result_dict and 'last_index' in result_dict:
        return result_dict['last_index']
    else:
        return 0
    
def fetch_etl_batch():
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # STEP 1: Check if etl_logs table is empty
    cursor.execute("SELECT COUNT(*) AS cnt FROM etl_logs")
    count = cursor.fetchone()['cnt']

    # --------------------------------------
    # CASE A: etl_logs is empty
    # --------------------------------------
    if count == 0:
        starting_index=1

    else:
        # --------------------------------------
        # CASE B: etl_logs already has entries
        # --------------------------------------
        cursor.execute("""
            SELECT last_updated_row 
            FROM etl_logs 
            ORDER BY last_updated_row DESC 
            LIMIT 1
        """)
        starting_index = cursor.fetchone()['last_updated_row']+1

    ending_row=starting_index+19999

    # STEP 2: Fetch raw_data for selected 7-day range
    
    query = """
        SELECT *
        FROM raw_data
        WHERE index_no BETWEEN %s AND %s
    """
    start_time=datetime.datetime.now()
    df = pd.read_sql(query, conn, params=[starting_index, ending_row])
    end_time=datetime.datetime.now()
    
    table_name='raw_data'
    length=int(df.shape[0])
    operation='Read'
    add_log(start_time,end_time,length,ending_row,table_name,operation)

    # log_query=f'''INSERT INTO etl_logs
    # (start_time,end_time,rows_processed, start_date,end_date,table_name,comments)
    # VALUES ({start_time},{end_time},{int(df.shape[0])},{start_date},{last_date},{table_name},"Read");'''
    # cursor.execute(log_query)
    cursor.close()
    conn.close()
    # Replace empty or 'NULL'-like strings with actual NaN in object columns
    df = df.replace({'': pd.NA, ' ': pd.NA, 'NULL': pd.NA, 'null': pd.NA})

    df['CustomerID'] = df['CustomerID'].fillna(0)
    last_row=int(df['index_no'].max())
    return df, last_row

# --- Classification of invalid rows based on Description ---
def classify_issue(description):
    if pd.isnull(description):
        return "Unclassified"
    
    desc = description.lower()
    if any(keyword in desc for keyword in ["adjust", "adjustment", "fee"]):
        return "Adjust"
    elif any(keyword in desc for keyword in ["wet", "damaged"]):
        return "Product Issue"
    else:
        return "Unclassified"


def routing_data(df):
   # Checking for not null 

   cond_not_null = df.notnull().all(axis=1)   
   
   # UnitPrice is non-negative
   cond_non_negative = df['UnitPrice'] >= 0

   # Combined condition
   valid_rows = cond_not_null & cond_non_negative
   valid_df = df[valid_rows].copy()      # rows meeting both conditions
   invalid_df = df[~valid_rows].copy()   # rows that don't
   
   # Create a new column
   invalid_df["Category"] = invalid_df["Description"].apply(classify_issue)
   
   cleaned_orders=valid_df[['index_no','InvoiceNo', 'StockCode', 'Description', 'Quantity',
      'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']].copy()
   special_cases=invalid_df[['index_no','InvoiceNo', 'StockCode', 'Description', 'Quantity',
      'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country', 'Category']].copy()
   special_cases = special_cases.replace({pd.NA: None, np.nan: None})
   
   conn = get_connection()
   cursor = conn.cursor(dictionary=True)

   
   insert_query1 = """
      INSERT INTO cleaned_orders
      (index_no,InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
      """

   start_time=datetime.datetime.now()
   for _, row in cleaned_orders.iterrows():
      cursor.execute(insert_query1,tuple(row))
   end_time=datetime.datetime.now()
   
   length=int(cleaned_orders.shape[0])
   ending_row=int(df['index_no'].max())
   table_name='cleaned_orders'
   operation='update'
   
   add_log(start_time,end_time,length,ending_row,table_name,operation)
   insert_query2 = """
      INSERT INTO special_cases
      (index_no,InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country,Type)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
   """
   start_time=datetime.datetime.now()
   for _, row in special_cases.iterrows():
      cursor.execute(insert_query2,tuple(row))
   end_time=datetime.datetime.now()
      
   length=int(special_cases.shape[0])
   table_name='special_cases'
   operation='update'
   add_log(start_time,end_time,length,ending_row,table_name,operation)
   
   conn.commit()
   cursor.close()
   return

def calculate_rfm(df):
    df = df.copy()

    # --- Clean data ---
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

    # Remove placeholder/anonymous customers
    df = df[df['CustomerID'] != 0]

    # Remove cancelled transactions (InvoiceNo starts with 'C' or negative Quantity)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]

    # Calculate total per line
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # Define snapshot date
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # --- RFM Calculation ---
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                   # Frequency
        'TotalAmount': 'sum'                                      # Monetary
    }).reset_index()

    # Rename columns
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)

    return rfm


def kmeans_rfm_segmentation(rfm_df, n_clusters=3, model_path='rfm_kmeans_model.pkl'):
    
    df = rfm_df.copy()

    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_features['Monetary'] = np.log1p(rfm_features['Monetary'])
    rfm_features['Frequency'] = np.log1p(rfm_features['Frequency'])
    rfm_features['Recency'] = np.log1p(rfm_features['Recency'])
    
    for col in ['Recency', 'Frequency', 'Monetary']:
        q_low, q_high = rfm_df[col].quantile([0.01, 0.99])
        rfm_df[col] = rfm_df[col].clip(lower=q_low, upper=q_high)


    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    joblib.dump(scaler,'StandardScaler.save')

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    joblib.dump({'model': kmeans, 'scaler': scaler}, model_path)

    cluster_summary = df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    cluster_summary['Segment'] = cluster_summary.apply(lambda row: (
        "High_value_customers" if (row['Monetary'] > cluster_summary['Monetary'].median() and 
                           row['Frequency'] > cluster_summary['Frequency'].median() and
                           row['Recency'] < cluster_summary['Recency'].median())
        else "Churned" if (row['Recency'] > cluster_summary['Recency'].median() and
                           row['Frequency'] < cluster_summary['Frequency'].median())
        else "Medium_Spender"
    ), axis=1)

    # --- Step 6: Merge cluster labels back ---
    segment_map = cluster_summary.set_index('Cluster')['Segment'].to_dict()
    df['Segment'] = df['Cluster'].map(segment_map)

    return df, cluster_summary

def customer_rfm_updation(cust_rfm):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    insert_query="""
INSERT INTO customer_rfm
(CustomerID,Recency,Frequency,Monetary,Cluster)
VALUES(%s, %s, %s, %s, %s);
"""
    start_time=datetime.datetime.now()
    for _, row in cust_rfm.iterrows():
        cursor.execute(insert_query,tuple(row))
    end_time=datetime.datetime.now()
    
    length=int(cust_rfm.shape[0])
    ending_row=length
    table_name='customer_rfm'
    operation='update'

    add_log(start_time,end_time,length,ending_row,table_name,operation)
    conn.commit()
    cursor.close()
    
    
    def getCleanedOrderData():
    get_rfm_query = 'SELECT * FROM cleaned_orders;'
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(get_rfm_query)
    rfm_data=cursor.fetchall()
    rfm_data1=pd.DataFrame(rfm_data)
    rfm_data1.columns=['index_no','InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']
    cursor.close()
    conn.close()
    return rfm_data1


if __name__ == "__main__":
    
    last_index=getLastIndex()
    while(True):
        df, x= fetch_etl_batch()
        routing_data(df)
        print('\a')
        if x==last_index:
            break
    rfm_data=calculate_rfm(getCleanedOrderData())
    Clustered_data=kmeans_rfm_segmentation(rfm_data)
    rfm_data1=Clustered_data[0][["CustomerID","Recency","Frequency","Monetary","Segment"]].copy()
    customer_rfm_updation(rfm_data1)