# DataAnalyzer

DataAnalyzer is a Python-based toolkit designed to simplify the process of e-commerce data analysis. It provides a streamlined interface to generate synthetic data, perform market basket analysis, conduct customer segmentation, and visualize results efficiently. The tool leverages powerful Python libraries to offer a comprehensive suite of features for data generation, analysis, and visualization.

## Features

- **Synthetic Data Generation**: Create realistic e-commerce transaction data for testing and analysis.
- **Market Basket Analysis**: Implement the Apriori algorithm to discover association rules in transaction data.
- **Customer Segmentation**: Utilize K-means clustering to segment customers based on their purchasing behavior.
- **Interactive Visualizations**: Generate 3D visualizations of association rules and customer clusters using Plotly.
- **Progress Tracking**: Monitor the progress of time-consuming operations like K-means clustering.
- **Customizable Analysis**: Tailor the analysis process with adjustable parameters for support, confidence, and number of clusters.

## Getting Started

To get started with DataAnalyzer, ensure you have Python 3.x installed on your system.

### Installation

Install the required dependencies:
```
pip install -r requirements.txt
```

### Usage

Run the main script to perform the complete analysis:

```python
python analyzer.py
```

This will generate synthetic data, perform Apriori algorithm for association rules, conduct K-means clustering for customer segmentation, and save visualizations as HTML files.

## Dependencies

DataAnalyzer relies on the following Python libraries:

- `numpy`: For numerical computations
- `pandas`: For data manipulation and analysis
- `faker`: For generating realistic product names
- `scikit-learn`: For K-means clustering and data preprocessing
- `plotly`: For interactive 3D visualizations
- `tqdm`: For progress bars

## Output

The script generates several HTML files:

- `apriori_rules_3d.html`: 3D visualization of top association rules
- `customer_clusters_3d_step_*.html`: Intermediate visualizations of customer clusters
- `customer_clusters_3d_final.html`: Final visualization of customer clusters

## Customization

You can customize the analysis by modifying parameters in the `main()` function:

- Adjust `num_products`, `num_customers`, and `num_transactions` to change the size of the synthetic dataset
- Modify `min_support` and `min_confidence` to alter the criteria for association rules
- Change `n_clusters` to adjust the number of customer segments

## Acknowledgments

- Inspiration for this project came from the need to simplify e-commerce data analysis tasks.
- Special thanks to the Python community for providing excellent libraries and resources.
