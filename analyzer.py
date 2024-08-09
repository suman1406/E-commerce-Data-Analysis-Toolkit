import numpy as np
import pandas as pd
from faker import Faker
import random
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Initialize Faker for generating realistic product names
fake = Faker()

def generate_data(num_products=10, num_customers=100, num_transactions=500):
    products = [fake.word() for _ in range(num_products)]
    transactions = []
    for _ in range(num_transactions):
        customer_id = random.randint(1, num_customers)
        basket_size = random.randint(1, 5)
        basket = random.sample(products, basket_size)
        transactions.append({
            'customer_id': customer_id,
            'products': basket
        })
    
    df = pd.DataFrame(transactions)
    df_encoded = df.explode('products').pivot_table(
        index='customer_id', 
        columns='products', 
        aggfunc=lambda x: 1, 
        fill_value=0
    )
    
    return df_encoded

def simple_apriori(df, min_support=0.1, min_confidence=0.5):
    def support(item_set):
        return (df[list(item_set)].sum(axis=1) == len(item_set)).mean()

    items = set(df.columns)
    item_sets = [frozenset([item]) for item in items]
    rules = []

    for k in range(2, len(items) + 1):
        item_sets = [s for s in combinations(items, k) if support(s) >= min_support]
        
        for item_set in item_sets:
            item_set = frozenset(item_set)
            for i in range(1, len(item_set)):
                for antecedent in combinations(item_set, i):
                    antecedent = frozenset(antecedent)
                    consequent = item_set - antecedent
                    confidence = support(item_set) / support(antecedent)
                    if confidence >= min_confidence:
                        lift = confidence / support(consequent)
                        rules.append({
                            'antecedents': ', '.join(antecedent),
                            'consequents': ', '.join(consequent),
                            'support': support(item_set),
                            'confidence': confidence,
                            'lift': lift
                        })
        
        if len(rules) >= 10:  # Stop if we have at least 10 rules
            break

    return pd.DataFrame(rules).sort_values('lift', ascending=False)

def perform_kmeans_with_progress(df, n_clusters=3, update_interval=5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
    
    with tqdm(total=kmeans.max_iter, desc="K-means Clustering") as pbar:
        for i in range(kmeans.max_iter):
            kmeans.fit(df_scaled)
            pbar.update(1)
            if i % update_interval == 0:
                yield kmeans.labels_
            if kmeans.n_iter_ <= i + 1:
                break
    
    return kmeans.labels_

def visualize_apriori_rules(rules, top_n=10):
    top_rules = rules.head(top_n)
    
    fig = px.scatter_3d(
        top_rules, x="support", y="confidence", z="lift",
        color="lift", size="support",
        hover_name="antecedents", hover_data=["consequents"],
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        title=f"Top {top_n} Association Rules"
    )
    
    return fig

def visualize_kmeans_clusters(df, cluster_labels):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df)
    
    fig = px.scatter_3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        color=cluster_labels,
        labels={'color': 'Cluster'},
        title="Customer Clusters Visualization"
    )
    
    return fig

def main():
    print("Generating synthetic e-commerce data...")
    df_encoded = generate_data(num_products=10, num_customers=100, num_transactions=500)
    print("Data generation complete.")
    print(f"Dataset shape: {df_encoded.shape}")

    print("Performing Apriori algorithm...")
    rules = simple_apriori(df_encoded, min_support=0.1, min_confidence=0.5)
    
    if not rules.empty:
        print(f"Apriori algorithm complete. Found {len(rules)} rules.")
        viz = visualize_apriori_rules(rules)
        viz.write_html("apriori_rules_3d.html")
        print("Apriori rules visualization saved as 'apriori_rules_3d.html'.")
    else:
        print("Apriori algorithm failed to generate rules.")

    print("Performing K-means clustering...")
    kmeans_generator = perform_kmeans_with_progress(df_encoded, n_clusters=3, update_interval=5)
    for i, labels in enumerate(kmeans_generator):
        print(f"K-means iteration {i*5}")
        viz = visualize_kmeans_clusters(df_encoded, labels)
        viz.write_html(f"customer_clusters_3d_step_{i}.html")
        print(f"Intermediate visualization saved as 'customer_clusters_3d_step_{i}.html'")
    
    final_labels = labels  # The last generated labels
    print("K-means clustering complete.")
    final_viz = visualize_kmeans_clusters(df_encoded, final_labels)
    final_viz.write_html("customer_clusters_3d_final.html")
    print("Final customer clusters visual saved as 'customer_clusters_3d_final.html'.")

    print("Analysis complete. ")

if __name__ == "__main__":
    main()
