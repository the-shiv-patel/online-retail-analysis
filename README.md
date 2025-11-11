---

# 🧾 Retail Sales Analytics – ETL + Power BI Dashboard

## 📘 Overview

This project performs **end-to-end data analysis** of retail sales data — from raw extraction to interactive dashboards in Power BI.
The goal is to monitor sales performance, customer behavior, and business health using automated ETL, RFM segmentation, and clustering.




**Flow Overview**

<img width="1172" height="1908" alt="High Level System Architecture Diagram (1)" src="https://github.com/user-attachments/assets/1a0efa02-1a9c-490f-8cd5-e8d07659473d" />

---

## 🏗️ DataBase Structure and Schema
<img width="1121" height="745" alt="yloy" src="https://github.com/user-attachments/assets/373999e0-f8e3-4619-9053-75658f0f565a" />

**ETL Highlights**

* Batch load from `raw_data` in chunks of 20,000 rows
* Data validation: non-null, non-negative prices
* Invalid rows routed to `special_cases`
* Automatic tagging for special records (`Adjustments`, `Product Issues`, `Unclassified`)
* ETL metadata logged in `etl_logs`

---

## 🧹 Data Processing

| Step                      | Description                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| **Data Extraction**       | Python fetches batches from MySQL table `raw_data`                                       |
| **Cleaning Rules**        | Filters invalid records and separates them                                               |
| **Transformation**        | Calculates new fields like total revenue, weekend/weekday flag                           |
| **Customer Segmentation** | Computes RFM (Recency, Frequency, Monetary) values                                       |
| **Clustering**            | Applies K-Means to segment customers into `High Value`, `Medium Spenders`, and `Churned` |
| **Storage**               | Final cleaned data stored in `cleaned_orders`, segmentation in `customer_rfm`            |

---

## 💾 Database Tables

| Table            | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| `raw_data`       | Original imported data from CSV                               |
| `cleaned_orders` | Valid, cleaned transactional records                          |
| `special_cases`  | Invalid or special entries with category labels               |
| `customer_rfm`   | Customer-level RFM metrics + cluster labels                   |
| `etl_logs`       | Tracks batch ETL operations (start/end times, rows processed) |

---

## 📊 Power BI Dashboards

### **1️⃣ Sales Dashboard**

**Focus:** Revenue trends, order volumes, top products, and KPIs
**Highlights:**

* Total Revenue, Orders, AOV, and Customer Count
* Revenue & Orders by Month
* Return Orders Trend
* Top 5 Products by Quantity & Revenue
* Optional: Revenue by Weekday or Country

### **2️⃣ Customer Dashboard**

**Focus:** Customer segmentation, retention, and value contribution
**Highlights:**

* Churn Count, Churn Rate, Avg Lifespan, CLV
* Segment Distribution (High Value, Medium, Churned)
* Average RFM by Segment
* Revenue Share per Segment
* RFM Scatterplot & Trend Over Time

---

## 🧠 Machine Learning Component

* **Algorithm:** K-Means Clustering
* **Features:** R, F, M scores (log-transformed + clipped for outliers)
* **Purpose:** Identify meaningful customer segments for retention and revenue targeting
* **Output:** Cluster label stored in `customer_rfm` table

---

## ⚙️ Tech Stack

| Layer                         | Tools                                     |
| ----------------------------- | ----------------------------------------- |
| **Storage**                   | MySQL                                     |
| **ETL & Processing**          | Python (pandas, mysql.connector, sklearn) |
| **Analytics & Visualization** | Power BI                                  |
| **Versioning & Docs**         | Git, Markdown                             |

---

## 🚀 Future Improvements

* Automate daily/weekly ETL refresh
* Add customer-level trend view in Power BI
* Build Power BI alerts for revenue drops or churn spikes
* Introduce more robust outlier detection beyond simple clipping



---

## ✨ Author

**[Shiv Patel]**
📧 [shiv.ds2004@gmail.com]
📊 Passionate about data, analytics, and automation.

