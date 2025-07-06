import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc, balanced_accuracy_score
)
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load your trained model and prepare test data"""
    # Load test data
    df = pd.read_csv('merged_data/complete_merged_dataset.csv')
    
    # Recreate feature engineering (same as training)
    df['Material_Wool_Flag'] = (df['Material'] == 'Wool').astype(int)
    df['Material_Cotton_Flag'] = (df['Material'] == 'Cotton').astype(int)
    df['Material_Polyester_Flag'] = (df['Material'] == 'Polyester').astype(int)
    df['Wool_Winter'] = df['Material_Wool_Flag'] * df['IsWinter']
    df['Heavy_Winter'] = (df['Weight'] > df['Weight'].median()).astype(int) * df['IsWinter']
    df['HighRisk_Supplier_Winter'] = (df['AvgBadPackagingRate'] > df['AvgBadPackagingRate'].median()).astype(int) * df['IsWinter']

    df['ProductRiskScore'] = (
        (df['Weight'] > df['Weight'].quantile(0.75)).astype(int) * 2 +
        df['Material_Wool_Flag'] * 2 +
        (df['ProductIncidentCount'] > 0).astype(int) * 1
    )
    df['SupplierRiskScore'] = (
        (df['AvgBadPackagingRate'] > df['AvgBadPackagingRate'].median()).astype(int) * 3 +
        (df['AvgOnTimeDeliveryRate'] < df['AvgOnTimeDeliveryRate'].median()).astype(int) * 2 +
        (df['SupplierIncidentCount'] > df['SupplierIncidentCount'].median()).astype(int) * 1
    )

    df['WeightCategory'] = pd.cut(df['Weight'], bins=3, labels=['Light', 'Medium', 'Heavy'])
    df['UnitsPerCartonCategory'] = pd.cut(df['ProposedUnitsPerCarton'], bins=3, labels=['Low', 'Medium', 'High'])

    # Recreate label encoders
    encoders = {}
    for col in ['ProposedFoldingMethod', 'ProposedLayout', 'GarmentType']:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
        df[f'{col}_encoded'] = le.transform(df[col].astype(str))
    
    # Weight and units categories
    le_weight = LabelEncoder()
    le_weight.fit(df['WeightCategory'].astype(str))
    df['WeightCategory_encoded'] = le_weight.transform(df['WeightCategory'].astype(str))
    
    le_units = LabelEncoder()
    le_units.fit(df['UnitsPerCartonCategory'].astype(str))
    df['UnitsPerCartonCategory_encoded'] = le_units.transform(df['UnitsPerCartonCategory'].astype(str))

    # Select features
    features = [
        'ProposedUnitsPerCarton', 'Weight',
        'Material_Wool_Flag', 'Material_Cotton_Flag', 'Material_Polyester_Flag',
        'ProductRiskScore', 'ProductIncidentCount', 'ProductTotalCost',
        'AvgBadPackagingRate', 'AvgOnTimeDeliveryRate', 'SupplierRiskScore',
        'SupplierIncidentCount', 'TotalPackagesHandled',
        'IsWinter', 'IsSummer', 'IsSpring', 'IsAutumn', 'IsPeakSeason', 'Month', 'Quarter',
        'Wool_Winter', 'Heavy_Winter', 'HighRisk_Supplier_Winter',
        'ProposedFoldingMethod_encoded', 'ProposedLayout_encoded', 'GarmentType_encoded',
        'WeightCategory_encoded', 'UnitsPerCartonCategory_encoded'
    ]
    features = [f for f in features if f in df.columns]
    
    X_test = df[features]
    y_test = df['PackagingQuality']
    
    # Load model
    model = joblib.load('model_outputs/best_model_xgboost.pkl')
    
    return model, X_test, y_test

def find_optimal_threshold(model, X_test, y_test, metric='f1'):
    """
    Find optimal threshold based on different metrics
    
    Parameters:
    - metric: 'f1', 'balanced_accuracy', 'bad_recall', 'bad_precision'
    """
    print(f"Finding optimal threshold based on {metric}...")
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of "Good" class
    
    # Test different thresholds
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    results = []
    
    for threshold in thresholds:
        # Make predictions with this threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        # Overall metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Class-specific metrics (Bad = 0, Good = 1)
        precision_bad = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        recall_bad = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_bad = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'bad_precision': precision_bad,
            'bad_recall': recall_bad,
            'bad_f1': f1_bad
        })
    
    results_df = pd.DataFrame(results)
    
    # Find best threshold based on chosen metric
    if metric == 'f1':
        best_idx = results_df['f1'].idxmax()
    elif metric == 'balanced_accuracy':
        best_idx = results_df['balanced_accuracy'].idxmax()
    elif metric == 'bad_recall':
        best_idx = results_df['bad_recall'].idxmax()
    elif metric == 'bad_precision':
        best_idx = results_df['bad_precision'].idxmax()
    elif metric == 'bad_f1':
        best_idx = results_df['bad_f1'].idxmax()
    else:
        best_idx = results_df['f1'].idxmax()  # Default to F1
    
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_metrics = results_df.loc[best_idx]
    
    print(f"\nBest threshold: {best_threshold:.3f}")
    print(f"Metrics at best threshold:")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"  Overall F1: {best_metrics['f1']:.4f}")
    print(f"  Bad Class Recall: {best_metrics['bad_recall']:.4f}")
    print(f"  Bad Class Precision: {best_metrics['bad_precision']:.4f}")
    print(f"  Bad Class F1: {best_metrics['bad_f1']:.4f}")
    
    return best_threshold, results_df

def evaluate_with_threshold(model, X_test, y_test, threshold):
    """Evaluate model performance with custom threshold"""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nConfusion Matrix (threshold = {threshold:.3f}):")
    print("               Predicted")
    print("               Bad    Good")
    print(f"Actual Bad   {cm[0,0]:6d} {cm[0,1]:7d}")
    print(f"Actual Good  {cm[1,0]:6d} {cm[1,1]:7d}")
    
    # Detailed metrics
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))
    
    # Business impact analysis
    total_bad = cm[0,0] + cm[0,1]  # Total actual bad cases
    total_good = cm[1,0] + cm[1,1]  # Total actual good cases
    
    bad_caught = cm[0,0]  # True negatives (bad correctly identified)
    bad_missed = cm[0,1]  # False positives (bad labeled as good)
    good_flagged = cm[1,0]  # False negatives (good labeled as bad)
    
    print(f"\nBusiness Impact Analysis:")
    print(f"  Bad packages detected: {bad_caught:,} out of {total_bad:,} ({bad_caught/total_bad*100:.1f}%)")
    print(f"  Bad packages missed: {bad_missed:,} ({bad_missed/total_bad*100:.1f}%)")
    print(f"  Good packages incorrectly flagged: {good_flagged:,} ({good_flagged/total_good*100:.1f}%)")
    
    return y_pred, y_proba

def plot_threshold_analysis(results_df):
    """Create visualizations for threshold analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall metrics vs threshold
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', alpha=0.8)
    ax1.plot(results_df['threshold'], results_df['balanced_accuracy'], label='Balanced Accuracy', alpha=0.8)
    ax1.plot(results_df['threshold'], results_df['f1'], label='F1 Score', alpha=0.8)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision and Recall vs threshold
    ax2 = axes[0, 1]
    ax2.plot(results_df['threshold'], results_df['precision'], label='Precision (Good)', alpha=0.8)
    ax2.plot(results_df['threshold'], results_df['recall'], label='Recall (Good)', alpha=0.8)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision/Recall vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bad class metrics vs threshold
    ax3 = axes[1, 0]
    ax3.plot(results_df['threshold'], results_df['bad_precision'], label='Bad Precision', alpha=0.8)
    ax3.plot(results_df['threshold'], results_df['bad_recall'], label='Bad Recall', alpha=0.8)
    ax3.plot(results_df['threshold'], results_df['bad_f1'], label='Bad F1', alpha=0.8)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('Bad Class Metrics vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trade-off analysis
    ax4 = axes[1, 1]
    ax4.plot(results_df['bad_recall'], results_df['bad_precision'], alpha=0.8)
    ax4.set_xlabel('Bad Recall (Sensitivity)')
    ax4.set_ylabel('Bad Precision')
    ax4.set_title('Precision-Recall Trade-off for Bad Class')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_thresholds(model, X_test, y_test):
    """Compare different threshold strategies"""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Define different threshold strategies
    strategies = {
        'Default (0.5)': 0.5,
        'Balanced Accuracy': None,  # Will be determined
        'Bad Class F1': None,
        'High Bad Recall (0.8+)': None,
        'Conservative (0.3)': 0.3
    }
    
    print("THRESHOLD STRATEGY COMPARISON")
    print("=" * 60)
    
    for strategy, threshold in strategies.items():
        if threshold is None:
            if strategy == 'Balanced Accuracy':
                _, results_df = find_optimal_threshold(model, X_test, y_test, 'balanced_accuracy')
                threshold = results_df.loc[results_df['balanced_accuracy'].idxmax(), 'threshold']
            elif strategy == 'Bad Class F1':
                _, results_df = find_optimal_threshold(model, X_test, y_test, 'bad_f1')
                threshold = results_df.loc[results_df['bad_f1'].idxmax(), 'threshold']
            elif strategy == 'High Bad Recall (0.8+)':
                _, results_df = find_optimal_threshold(model, X_test, y_test, 'bad_recall')
                # Find threshold that gives bad recall >= 0.8
                high_recall = results_df[results_df['bad_recall'] >= 0.8]
                if len(high_recall) > 0:
                    threshold = high_recall.loc[high_recall['bad_precision'].idxmax(), 'threshold']
                else:
                    threshold = results_df.loc[results_df['bad_recall'].idxmax(), 'threshold']
        
        print(f"\n{strategy} (threshold = {threshold:.3f}):")
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        bad_recall = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        bad_precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"  Bad packages caught: {cm[0,0]:,} / {cm[0,0] + cm[0,1]:,} ({bad_recall*100:.1f}%)")
        print(f"  False alarms: {cm[1,0]:,} / {cm[1,0] + cm[1,1]:,} ({cm[1,0]/(cm[1,0] + cm[1,1])*100:.1f}%)")
        print(f"  Balanced Accuracy: {balanced_acc:.3f}")

def main_threshold_tuning():
    """Main function to run threshold tuning analysis"""
    print("THRESHOLD TUNING FOR PACKAGING QUALITY MODEL")
    print("=" * 50)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Find optimal thresholds for different metrics
    print("\n1. Finding optimal threshold for balanced accuracy...")
    best_threshold_bal, results_df = find_optimal_threshold(model, X_test, y_test, 'balanced_accuracy')
    
    print("\n2. Finding optimal threshold for bad class F1...")
    best_threshold_bad_f1, _ = find_optimal_threshold(model, X_test, y_test, 'bad_f1')
    
    # Evaluate with best threshold
    print(f"\n3. Detailed evaluation with balanced accuracy threshold ({best_threshold_bal:.3f}):")
    evaluate_with_threshold(model, X_test, y_test, best_threshold_bal)
    
    # Create visualizations
    print("\n4. Creating threshold analysis plots...")
    plot_threshold_analysis(results_df)
    
    # Compare different strategies
    print("\n5. Comparing different threshold strategies...")
    compare_thresholds(model, X_test, y_test)
    
    # Recommendations
    print(f"\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("=" * 50)
    print(f"• For best overall balance: Use threshold = {best_threshold_bal:.3f}")
    print(f"• For better bad class detection: Use threshold = {best_threshold_bad_f1:.3f}")
    print("• Consider business costs when choosing final threshold")
    print("• Monitor performance in production and adjust as needed")
    
    return best_threshold_bal, best_threshold_bad_f1, results_df

if __name__ == "__main__":
    best_bal_thresh, best_bad_f1_thresh, results = main_threshold_tuning()