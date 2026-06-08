
# 🛍️ Customer Segmentation — Online Retail
 
> Unsupervised Machine Learning · RFM Analysis · K-Means Clustering · LLM-Powered Recommendations
 
---
 
## 📌 Overview
 
This project applies **data-driven customer segmentation** to a real-world UK online retail dataset. Rather than treating all customers the same, the goal is to identify distinct behavioural profiles — each with its own purchasing patterns, value, and engagement level — and translate them into actionable marketing strategies.
 
A **Llama 3.3-70B** recommendation system is also integrated at the end, generating personalised product suggestions for each discovered segment.
 
---
 
## 📦 Dataset
 
**UCI Online Retail Dataset** — transactional records from a UK-based non-store online retailer specialising in gift and home décor products.
 
| Field | Description |
|:---|:---|
| `InvoiceNo` | Unique transaction ID (prefix `C` = cancellation) |
| `StockCode` | Unique product identifier |
| `Description` | Product name |
| `Quantity` | Units purchased per line |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit (GBP £) |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer country |
 
- **Period:** December 2010 – December 2011
- **Scale:** 540,000+ transactions across 38 countries
- **UK share:** ~91% of all activity
---
 
## 🎯 Objectives
 
1. Engineer rich customer-level features beyond standard RFM (Recency, Frequency, Monetary)
2. Clean and prepare the data rigorously — handling cancellations, anomalous stock codes, and zero-price transactions
3. Apply unsupervised clustering on PCA-reduced features to identify stable customer segments
4. Interpret each segment as an actionable business persona with strategic recommendations
5. Generate personalised product recommendations per segment using a large language model
---
 
## 🔬 Methodology
 
| Step | Technique | Purpose |
|:---|:---|:---|
| EDA | Univariate · Bivariate · Multivariate · Domain-specific | Understand distributions, correlations, and business dynamics |
| Data Cleaning | Deduplication · Imputation · Filtering | Ensure only valid, attributable transactions remain |
| Feature Engineering | RFM + 7 extended features | Build a 360° behavioural profile per customer |
| Outlier Treatment | Isolation Forest (5% contamination) | Remove anomalous profiles that would distort cluster centroids |
| Scaling + PCA | StandardScaler · PCA (80% variance retained) | Normalise magnitudes and eliminate multicollinearity |
| Clustering | K-Means++ · Elbow · Silhouette analysis | Discover and validate the optimal number of segments |
| Evaluation | Silhouette · Calinski-Harabász · Davies-Bouldin | Quantify cluster cohesion and separation |
| Recommendations | Llama 3.3-70B via Groq API (zero-shot) | Personalised product suggestions per segment |
 
---
 
## 🧠 Engineered Features
 
| Feature | Description |
|:---|:---|
| `Recency` | Days since last purchase |
| `Total_orders` | Number of unique transactions |
| `Total_Purchase` | Total units bought |
| `Total_Spend` | Total revenue contributed |
| `Average_Transaction_Value` | Mean spend per order |
| `Is_UK` | Binary domestic/international flag |
| `Customer_Lifetime` | Days between first and last purchase |
| `Daily_Customer_Value` | Total spend ÷ customer lifetime |
| `Average_Basket_Size` | Mean units per transaction |
| `Unique_Products_Purchased` | Number of distinct products bought |
| `Average_Days_Between_Purchases` | Mean inter-purchase interval |
| `Favorite_Shopping_Day` | Most frequent purchase day of week |
| `Favorite_Shopping_Hour` | Most frequent purchase hour |
 
---
 
## 📊 Results — Customer Segments
 
| Cluster | Label | Share | Profile |
|:---:|:---|:---:|:---|
| 0 | 🟣 **Loyal Mid-Value Customers** | ~44.6% | Recent, moderate frequency, healthy spend. The retailer's reliable core base. |
| 1 | 🔴 **At-Risk / Dormant Customers** | ~28.3% | High recency, very few orders, minimal spend. Prime targets for win-back campaigns. |
| 2 | 🩷 **Champions — Power Shoppers** | ~14.9% | Most recent, highest order count, largest spend, widest product variety. VIP segment. |
| 3 | 💜 **High-Value Occasional Buyers** | ~12.2% | Infrequent orders, very high transaction value. Likely B2B or bulk buyers. |
 
### Strategic Recommendations
 
- **Champions (Cluster 2):** Loyalty rewards, exclusive access, dedicated account management
- **Loyal Mid-Value (Cluster 0):** Upsell/cross-sell, personalised recommendations to nudge toward Champion status
- **High-Value Occasional (Cluster 3):** Seasonal re-engagement campaigns, reduce inter-purchase intervals
- **At-Risk (Cluster 1):** Urgent win-back offers; sunset or survey campaigns for long-dormant customers
---
 
## 🏗️ Project Structure
 
```
CustomerSegmentation.ipynb
│
├── 1. Dataset Overview
├── 2. Exploratory Data Analysis (EDA)
│   ├── 2.1 Univariate Analysis
│   ├── 2.2 Bivariate Analysis
│   ├── 2.3 Multivariate Analysis
│   └── 2.4 Domain-Specific Analysis (Geographic, Product, Cancellations)
├── 3. Data Cleaning
│   ├── 3.1 Duplicates
│   ├── 3.2 Missing Values
│   ├── 3.3 Data Transformation
│   └── 3.4 Anomalies (StockCode, Description, Zero Prices)
├── 4. Feature Engineering
│   ├── 4.1 RFM (Recency · Frequency · Monetary)
│   ├── 4.2 Geographic Features
│   ├── 4.3 Customer Lifetime
│   ├── 4.4 Daily Customer Value
│   ├── 4.5 Average Basket Size
│   ├── 4.6 Product Diversity
│   └── 4.7 Behavioural Features
├── 5. Outlier Detection (Isolation Forest)
├── 6. Feature Scaling (StandardScaler)
├── 7. Dimensionality Reduction (PCA)
├── 8. K-Means Clustering
│   ├── 8.1 Elbow + Silhouette Analysis
│   ├── 8.2 Final Model (k=4)
│   ├── 8.3 2D PCA Visualisation
│   ├── 8.4 Cluster Size Distribution
│   └── 8.5 Segment Profiles & Interpretation
├── 9. Model Evaluation Metrics
└── 10. LLM Recommendation System (Llama 3.3-70B via Groq)
```
 
---
 
## 🛠️ Tech Stack
 
- **Python 3**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn**, **plotly** — visualisation
- **scikit-learn** — Isolation Forest, StandardScaler, PCA, K-Means, evaluation metrics
- **yellowbrick** — Silhouette visualiser
- **Groq API** + **Llama 3.3-70B** — LLM-powered recommendations
---
 
