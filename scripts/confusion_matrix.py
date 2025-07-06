import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Model results data
results_data = {
    'Model': ['Logistic Regression (Default 0.5)', 'Logistic Regression (Optimal 0.86)', 
              'Random Forest (Default 0.5)', 'Random Forest (Optimal 0.86)', 
              'XGBoost (Default 0.5)', 'XGBoost (Optimal 0.86)'],
    'Threshold': [0.5, 0.86, 0.5, 0.86, 0.5, 0.86],
    'Accuracy': [0.6139, 0.1986, 0.6154, 0.1986, 0.8014, 0.5217],
    'Balanced_Accuracy': [0.6497, 0.5000, 0.6530, 0.5000, 0.5014, 0.6365],
    'Precision': [0.8912, 0.0, 0.8933, 0.0, 0.8018, 0.9123],
    'Recall': [0.5903, 0.0, 0.5906, 0.0, 0.9991, 0.4460],
    'F1': [0.7102, 0.0, 0.7111, 0.0, 0.8896, 0.5991],
    'AUC': [0.6934, 0.6934, 0.7000, 0.7000, 0.7007, 0.7007],
    'Bad_Precision': [0.3002, 0.1986, 0.3022, 0.1986, 0.4931, 0.2701],
    'Bad_Recall': [0.7092, 1.0000, 0.7155, 1.0000, 0.0037, 0.8271],
    'Bad_F1': [0.4218, 0.3314, 0.4249, 0.3314, 0.0073, 0.4072]
}

df = pd.DataFrame(results_data)

# Test set composition (from your output)
TOTAL_TEST = 97630
TOTAL_BAD = 19390
TOTAL_GOOD = 78240

def calculate_confusion_matrix(bad_recall, bad_precision, total_bad, total_good):
    """Calculate confusion matrix from recall and precision metrics"""
    
    # True Positives (Bad correctly identified as Bad)
    TP = int(bad_recall * total_bad)
    
    # False Negatives (Bad incorrectly identified as Good)
    FN = total_bad - TP
    
    # Calculate False Positives from precision
    # Precision = TP / (TP + FP)
    # So: FP = (TP / Precision) - TP
    if bad_precision > 0:
        FP = int((TP / bad_precision) - TP)
    else:
        FP = total_good  # If precision is 0, all good packages are flagged
    
    # True Negatives
    TN = total_good - FP
    
    return np.array([[TP, FN], [FP, TN]])

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Confusion Matrices: Model and Threshold Comparison', fontsize=16, fontweight='bold')

# Define colors for different models
colors = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'YlOrRd']

for i, (_, row) in enumerate(df.iterrows()):
    ax = axes[i//3, i%3]
    
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(
        row['Bad_Recall'], 
        row['Bad_Precision'], 
        TOTAL_BAD, 
        TOTAL_GOOD
    )
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i], 
                xticklabels=['Bad', 'Good'], 
                yticklabels=['Bad', 'Good'],
                cbar_kws={'label': 'Count'}, ax=ax)
    
    # Set title and labels
    model_name = row['Model'].replace(' (Default 0.5)', '').replace(' (Optimal 0.86)', '')
    threshold_type = 'Default (0.5)' if row['Threshold'] == 0.5 else 'Optimal (0.86)'
    
    ax.set_title(f'{model_name}\n{threshold_type}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    
    # Add performance metrics as text
    metrics_text = f"Bad Recall: {row['Bad_Recall']:.1%}\nBad Precision: {row['Bad_Precision']:.1%}\nAUC: {row['AUC']:.3f}"
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print detailed business impact for each model
print("\n" + "="*80)
print("BUSINESS IMPACT ANALYSIS BY MODEL")
print("="*80)

for _, row in df.iterrows():
    print(f"\n{row['Model']}:")
    
    cm = calculate_confusion_matrix(
        row['Bad_Recall'], 
        row['Bad_Precision'], 
        TOTAL_BAD, 
        TOTAL_GOOD
    )
    
    TP, FN = cm[0]  # Bad packages: True Positives, False Negatives
    FP, TN = cm[1]  # Good packages: False Positives, True Negatives
    
    print(f"  Bad packages correctly identified: {TP:,} out of {TOTAL_BAD:,} ({TP/TOTAL_BAD:.1%})")
    print(f"  Bad packages missed: {FN:,} ({FN/TOTAL_BAD:.1%})")
    print(f"  Good packages incorrectly flagged: {FP:,} out of {TOTAL_GOOD:,} ({FP/TOTAL_GOOD:.1%})")
    print(f"  Good packages correctly identified: {TN:,} ({TN/TOTAL_GOOD:.1%})")
    
    # Business calculations (using EUR)
    cost_per_incident = 580  # EUR from your supplier analysis
    cost_per_inspection = 5  # EUR
    
    savings = TP * cost_per_incident
    inspection_costs = FP * cost_per_inspection
    net_benefit = savings - inspection_costs
    
    print(f"  Savings from prevented incidents: €{savings:,}")
    print(f"  Cost of unnecessary inspections: €{inspection_costs:,}")
    print(f"  Net benefit: €{net_benefit:,}")
    if inspection_costs > 0:
        roi = (net_benefit / inspection_costs) * 100
        print(f"  ROI: {roi:.1f}%")
    else:
        print(f"  ROI: Infinite (no inspection costs)")

# Create a summary comparison table
print("\n" + "="*80)
print("SUMMARY COMPARISON TABLE")
print("="*80)

summary_data = []
for _, row in df.iterrows():
    cm = calculate_confusion_matrix(row['Bad_Recall'], row['Bad_Precision'], TOTAL_BAD, TOTAL_GOOD)
    TP, FN = cm[0]
    FP, TN = cm[1]
    
    savings = TP * 580
    costs = FP * 5
    net = savings - costs
    
    summary_data.append({
        'Model': row['Model'],
        'Bad_Detected': TP,
        'Bad_Missed': FN,
        'False_Alarms': FP,
        'Net_Benefit_EUR': net,
        'AUC': row['AUC']
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False, float_format='%.0f'))

# Highlight the best performing model
best_model_idx = summary_df['Net_Benefit_EUR'].idxmax()
best_model = summary_df.iloc[best_model_idx]

print(f"\n🏆 BEST PERFORMING MODEL:")
print(f"   {best_model['Model']}")
print(f"   Bad packages detected: {best_model['Bad_Detected']:,.0f}")
print(f"   Net benefit: €{best_model['Net_Benefit_EUR']:,.0f}")
print(f"   AUC: {best_model['AUC']:.3f}")