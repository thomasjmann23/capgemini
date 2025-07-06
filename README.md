# AI-Powered Packaging Quality Optimization
## Capstone Project for Fashion World Retail

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-Academic-orange.svg)](LICENSE)

**Client:** Capgemini × Fashion World Retail  
**Team:** Maria Vazquez Pinedo, Thomas Mann, Fatine Samodi, Nicole M. Batinovich, Ayush Singh  
**Supervisor:** Maria Victoria Rivas Lopez  
**Duration:** May 19 - July 11, 2025

---

## Executive Summary

This project develops an AI-powered solution to optimize packaging quality in the fashion retail supply chain. Using machine learning on 500,000+ historical packaging records, we achieved **82.7% detection of bad packaging incidents** while transforming reactive quality control into predictive quality management.

### Key Achievements
- **XGBoost model** with AUC-ROC of 0.701
- **82.7% bad package detection** (vs 0.4% baseline)
- **€9.3M potential annual savings** through incident prevention
- **Production-ready deployment** with interactive cost-benefit calculator

---

## Business Impact

### Problem Statement
Fashion World Retail faced critical packaging challenges:
- **Manual packaging decisions** leading to inconsistent quality
- **Fragmented data sources** preventing holistic analysis  
- **Seasonal fluctuations** causing packaging failures during peak periods
- **Supplier variability** affecting overall packaging quality

### Solution Benefits
- **Proactive Quality Control:** Predict and prevent packaging failures before shipment
- **Supplier Intelligence:** Data-driven insights for supplier performance optimization
- **Cost Reduction:** Significant savings through early problem detection
- **Scalable Architecture:** Enterprise-ready ML pipeline for production deployment

---

## Technical Architecture

### Data Pipeline
```
Raw Data Sources (4 datasets, 500k+ records)
    ↓
Data Cleaning & Standardization
    ↓  
Feature Engineering (28 final features)
    ↓
Model Training & Optimization
    ↓
Threshold Optimization (0.86 optimal)
    ↓
Production Deployment
```

### Machine Learning Pipeline
- **Algorithms:** Logistic Regression, Random Forest, XGBoost
- **Feature Engineering:** Domain-specific interactions (Wool×Winter, Supplier×Season)
- **Class Imbalance Handling:** Advanced techniques for 80/20 distribution
- **Threshold Optimization:** Business-focused optimization for maximum impact

---

## Repository Structure

```
├── 📄 Final_Report.pdf                    # Comprehensive project documentation
├── 📄 README.md                          # This file
├── 📄 cost_benefit_calculator.html       # Interactive threshold calculator
├── 📁 scripts/
│   ├── 📄 ml_pipeline_full.py            # Complete ML pipeline with hyperparameter tuning
│   ├── 📄 ml_pipeline_small.py           # Streamlined pipeline for testing
│   ├── 📄 threshold_tuning.py            # Threshold optimization analysis
│   └── 📄 confusion_matrix.py            # Detailed performance visualization
├── 📁 model_outputs/
│   ├── 📄 best_model_xgboost_optimized.pkl
│   ├── 📄 optimal_threshold_config.json
│   ├── 📄 comprehensive_model_results.csv
│   └── 📄 feature_list.txt
└── 📁 merged_data/
    └── 📄 complete_merged_dataset.csv     # Fully processed dataset (confidential)
```

---

## Installation & Setup

### Prerequisites
```bash
Python 3.9+
pandas >= 1.3.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/thomasjmann23/capgemini.git
cd capgemini

# Install dependencies
pip install -r requirements.txt

# Run streamlined pipeline (recommended for testing)
python scripts/ml_pipeline_small.py

# Run full pipeline with hyperparameter tuning
python scripts/ml_pipeline_full.py

# Perform threshold optimization
python scripts/threshold_tuning.py
```

---

## Model Performance

### Performance Metrics
| Model | Threshold | Bad Detection | Precision | AUC-ROC | Business Impact |
|-------|-----------|---------------|-----------|---------|-----------------|
| XGBoost (Default) | 0.5 | 0.4% | 49.3% | 0.701 | Baseline |
| **XGBoost (Optimized)** | **0.86** | **82.7%** | **27.0%** | **0.701** | **+15,966 packages** |
| Random Forest | 0.86 | 100% | 19.9% | 0.700 | High false alarms |

### Feature Importance (Top 5)
1. **AvgOnTimeDeliveryRate** (50.9%) - Supplier delivery consistency
2. **AvgBadPackagingRate** (29.9%) - Historical supplier performance  
3. **ProposedFoldingMethod** (6.3%) - Packaging technique
4. **Material_Wool_Flag** (1.9%) - Material complexity
5. **SupplierIncidentCount** (1.6%) - Historical incidents

---

## Business Intelligence Dashboard

### Interactive Cost-Benefit Calculator
Our solution includes a sophisticated web-based calculator that enables stakeholders to:
- **Model different threshold strategies** based on operational priorities
- **Calculate ROI** using custom cost parameters
- **Compare scenarios** from conservative to high-detection approaches
- **Optimize resource allocation** for quality inspection teams

**Access:** Open `cost_benefit_calculator.html` in your browser

### Threshold Strategies
- **Conservative (0.50):** Minimal false alarms, limited detection
- **F1-Optimized (0.79):** 69.2% detection, balanced approach  
- **Balanced (0.80):** 71.7% detection, operational sweet spot
- **High Detection (0.86):** 82.7% detection, maximum protection

---

## Production Deployment

### Model Deployment
```python
from model_outputs.deployment_script import PackagingQualityPredictor

# Initialize predictor
predictor = PackagingQualityPredictor()

# Make predictions
predictions, probabilities = predictor.predict(new_data)

# Get risk levels
risk_levels, probabilities = predictor.predict_risk_level(new_data)
```

### Monitoring & Maintenance
- **Real-time performance tracking** with automated alerts
- **Quarterly model retraining** to handle data drift
- **A/B testing framework** for threshold optimization
- **Supplier performance dashboards** for continuous improvement

---

## Methodology

### Data Processing Pipeline
1. **Data Integration:** 4 disparate datasets unified with 99.7% join success
2. **Quality Assurance:** Supplier name normalization (47 variants → 5 standardized)
3. **Feature Engineering:** Domain expertise captured in interaction terms
4. **Class Imbalance:** Advanced techniques for 80/20 Good/Bad distribution

### Model Development
1. **Algorithm Selection:** Systematic evaluation of 3 ML approaches
2. **Hyperparameter Tuning:** GridSearchCV with 3-fold cross-validation
3. **Threshold Optimization:** Business-focused optimization beyond standard 0.5
4. **Validation:** Comprehensive testing with production-realistic scenarios

---

## Results & Impact

### Quantitative Results
- **Detection Improvement:** 82.3% increase in bad package detection
- **Financial Impact:** €9.3M potential annual savings
- **Operational Efficiency:** Reduced manual decision-making
- **Supplier Intelligence:** Actionable insights for 8 suppliers

### Strategic Value
- **Competitive Advantage:** Industry-leading AI-driven quality control
- **Scalability:** Foundation for additional supply chain AI applications  
- **Risk Mitigation:** Proactive approach to quality management
- **Data-Driven Culture:** Enhanced decision-making across operations

---

## Research & Development

### Technical Innovations
1. **Threshold Optimization Framework:** Moving beyond standard 0.5 thresholds
2. **Advanced Class Imbalance Handling:** Maximizing minority class detection
3. **Domain-Knowledge Feature Engineering:** Fashion retail expertise integration
4. **Production-Ready Architecture:** Enterprise deployment considerations

### Future Enhancements
- **Multi-objective Optimization:** Sustainability goals integration
- **Real-time Adaptation:** Dynamic thresholds based on operational capacity
- **Expanded Coverage:** Additional product categories and seasonal collections
- **Advanced Analytics:** Predictive supplier risk scoring for procurement

---

## Team & Acknowledgments

**Student Team:**
- Maria Vazquez Pinedo
- Thomas Mann  
- Fatine Samodi
- Nicole M. Batinovich
- Ayush Singh

**Academic Supervisor:** Maria Victoria Rivas Lopez  
**Industry Partner:** Capgemini  
**Client:** Fashion World Retail

**Special Thanks:** Capgemini for providing real-world datasets and domain expertise that made this project possible.

---

## License & Usage

This project is developed for academic purposes in collaboration with Capgemini. The methodology and code structure are available for educational use. Proprietary data and specific business logic remain confidential.

**Contact:** [GitHub Repository](https://github.com/thomasjmann23/capgemini/tree/main)

---

*This project demonstrates the practical application of machine learning in enterprise supply chain optimization, delivering measurable business value through innovative AI solutions.*
