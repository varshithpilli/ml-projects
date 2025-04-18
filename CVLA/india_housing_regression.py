import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def inverse_gauss_jordan(A):
    A = A.astype(float)                        # Convert to float (decimal)
    n = A.shape[0]                             # No. of rows in A
    
    I = np.eye(n)                              # n * n identity matrix
    aug = np.hstack((A, I))                    # Horizontal stack (A & I) => [A|I]

    for i in range(n):
        # Ensure the pivot is non-zero. If it's zero, find a non-zero element and swap rows.
        if aug[i, i] == 0:
            for j in range(i + 1, n):
                if aug[j, i] != 0:
                    aug[[i, j]] = aug[[j, i]]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")
        
        # Make the diagonal element = 1 (for RREF)
        aug[i] = aug[i] / aug[i, i]
        
        # Normalise other rows using standard row operations
        for j in range(n):
            if j != i:
                aug[j] -= aug[j, i] * aug[i]

    # Get the inverse matrix from the augmented matrix
    A_inv = aug[:, n:]
    return A_inv


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.intercept = None
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # θ = (X^T*X)^(-1)*X^T*y
        XT_X = np.dot(X_b.T, X_b)
        XT_y = np.dot(X_b.T, y)
        
        XT_X_inv = inverse_gauss_jordan(XT_X)
        
        self.weights = np.dot(XT_X_inv, XT_y)

        self.intercept = self.weights[0]
        self.weights = self.weights[1:]
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

def print_flush(message):
    print(message)
    sys.stdout.flush()

def calculate_metrics(y_true, y_pred, prefix=""):
    metrics = {}

    # R²
    metrics["r2"] = r2_score(y_true, y_pred)
    
    # MAPE
    def mape(y_true, y_pred): 
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    metrics["mape"] = mape(y_true, y_pred)

    # Accuracy
    metrics["accuracy"] = 100 - metrics["mape"]
    
    # MSE
    metrics["mse"] = mean_squared_error(y_true, y_pred)

    # RMSE
    metrics["rmse"] = np.sqrt(metrics["mse"])

    # MAE
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    
    return metrics

def main():
    print_flush("=" * 63)
    print_flush("Linear Regression Model Comparison for Housing Price Prediction")
    print_flush("=" * 63)
    
    # Step - 1
    print_flush("\n1. Loading House Price India dataset...")
    data = pd.read_csv("House Price India.csv")
    print_flush(f"   - Dataset loaded successfully with {data.shape[0]} samples and {data.shape[1]} features")
    
    exclude_columns = ['id', 'Date', 'Postal Code', 'Price']
    feature_columns = [col for col in data.columns if col not in exclude_columns]
        
    X = data[feature_columns].values
    y = data['Price'].values
        
    print_flush(f"   - Selected {len(feature_columns)} features for modeling")
        
    # Step - 2
    print_flush("\n2. Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print_flush("   - Features standardized using StandardScaler")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=41)
    print_flush(f"   - Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Step - 3
    print_flush("\n3. Training and comparing multiple regression models...")
    
    models = {}
    predictions = {}
    metrics = {}
    
    print_flush("\n   3.1 Training custom linear regression model ...")
    scratch_model = LinearRegression()
    scratch_model.fit(X_train, y_train)
    models["scratch"] = scratch_model
    predictions["scratch_train"] = scratch_model.predict(X_train)
    predictions["scratch_test"] = scratch_model.predict(X_test)
    
    print_flush("   3.2 Training scikit-learn's LinearRegression model...")
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    models["sklearn"] = sklearn_model
    predictions["sklearn_train"] = sklearn_model.predict(X_train)
    predictions["sklearn_test"] = sklearn_model.predict(X_test)
    
    print_flush("   3.3 Training Ridge regression model (L2 regularization)...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    models["ridge"] = ridge_model
    predictions["ridge_train"] = ridge_model.predict(X_train)
    predictions["ridge_test"] = ridge_model.predict(X_test)
    
    print_flush("   3.4 Training Lasso regression model (L1 regularization)...")
    lasso_model = Lasso(alpha=1.0)
    lasso_model.fit(X_train, y_train)
    models["lasso"] = lasso_model
    predictions["lasso_train"] = lasso_model.predict(X_train)
    predictions["lasso_test"] = lasso_model.predict(X_test)
    
    # Step - 4
    print_flush("\n4. Model Performance Comparison:")
    
    def adjusted_r2(r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    metrics["scratch"] = calculate_metrics(y_test, predictions["scratch_test"])
    metrics["scratch"]["adj_r2"] = adjusted_r2(metrics["scratch"]["r2"], len(y_test), X_test.shape[1])
    
    metrics["sklearn"] = calculate_metrics(y_test, predictions["sklearn_test"])
    metrics["sklearn"]["adj_r2"] = adjusted_r2(metrics["sklearn"]["r2"], len(y_test), X_test.shape[1])
    
    metrics["ridge"] = calculate_metrics(y_test, predictions["ridge_test"])
    metrics["ridge"]["adj_r2"] = adjusted_r2(metrics["ridge"]["r2"], len(y_test), X_test.shape[1])
    
    metrics["lasso"] = calculate_metrics(y_test, predictions["lasso_test"])
    metrics["lasso"]["adj_r2"] = adjusted_r2(metrics["lasso"]["r2"], len(y_test), X_test.shape[1])
    
    print_flush("\n   Model Comparison on Test Set:")
    print_flush("   " + "-" * 89)
    print_flush("   | {:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |".format(
        "Model", "R²", "Adj. R²", "RMSE", "MAE", "Accuracy"
    ))
    print_flush("   " + "-" * 89)
    
    for model_name in ["scratch", "sklearn", "ridge", "lasso"]:
        print_flush("   | {:<20} | {:<10.4f} | {:<10.4f} | {:<10.2f} | {:<10.2f} | {:<10.2f} |".format(
            model_name.capitalize(),
            metrics[model_name]["r2"],
            metrics[model_name]["adj_r2"],
            metrics[model_name]["rmse"],
            metrics[model_name]["mae"],
            metrics[model_name]["accuracy"]
        ))
    
    print_flush("   " + "-" * 89)
    
    # Step - 5
    print_flush("\n5. Creating visualizations...")
    
    # Actual vs Predicted
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    model_names = ["scratch", "sklearn", "ridge", "lasso"]
    
    for i, model_name in enumerate(model_names):
        axes[i].scatter(y_test, predictions[f"{model_name}_test"], alpha=0.5)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes[i].set_xlabel('Actual Price ($)')
        axes[i].set_ylabel('Predicted Price ($)')
        axes[i].set_title(f'{model_name.capitalize()} Model: Actual vs Predicted (R² = {metrics[model_name]["r2"]:.4f})')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    print_flush("   - Model comparison plot saved as 'plots/model_comparison.png'")
    
    # Residuals
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        residuals = y_test - predictions[f"{model_name}_test"]
        axes[i].scatter(predictions[f"{model_name}_test"], residuals, alpha=0.5)
        axes[i].axhline(y=0, color='r', linestyle='-')
        axes[i].set_xlabel('Predicted Price ($)')
        axes[i].set_ylabel('Residuals ($)')
        axes[i].set_title(f'{model_name.capitalize()} Model: Residual Plot')
    
    plt.tight_layout()
    plt.savefig('plots/residual_comparison.png')
    print_flush("   - Residual comparison plot saved as 'plots/residual_comparison.png'")
    
    # Performance metrics
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.2
    
    # R²
    plt.bar(x - 1.5*width, [metrics[m]["r2"] for m in model_names], width, label='R²')
    # Accuracy
    plt.bar(x - 0.5*width, [metrics[m]["accuracy"]/100 for m in model_names], width, label='Accuracy (scaled)')
    # RMSE
    plt.bar(x + 0.5*width, [metrics[m]["rmse"]/1000000 for m in model_names], width, label='RMSE (scaled ÷ 1,000,000)')
    # MAE
    plt.bar(x + 1.5*width, [metrics[m]["mae"]/1000000 for m in model_names], width, label='MAE (scaled ÷ 1,000,000)')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, [m.capitalize() for m in model_names])
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/performance_comparison.png')
    print_flush("   - Performance comparison plot saved as 'plots/performance_comparison.png'")
    
    print_flush("\nModel comparison complete!")

if __name__ == "__main__":
    main()