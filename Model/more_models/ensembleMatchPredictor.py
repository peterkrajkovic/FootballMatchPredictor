import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EnsembleMatchPredictor:
    """Ensemble of Neural Networks with different architectures"""
    
    def __init__(self, input_size, device='cpu'):
        self.input_size = input_size
        self.device = device
        self.models = []
        self.scalers = []
        
        # Create diverse architectures
        architectures = [
            {'hidden_sizes': [128, 96, 64, 32], 'dropout': 0.3, 'name': 'deep'},
            {'hidden_sizes': [256, 128, 64], 'dropout': 0.2, 'name': 'wide'},
            {'hidden_sizes': [64, 128, 64, 32], 'dropout': 0.4, 'name': 'narrow'},
            {'hidden_sizes': [96, 96, 96], 'dropout': 0.25, 'name': 'uniform'},
        ]
        
        for arch in architectures:
            model = AdvancedMatchPredictor(
                input_size=input_size,
                hidden_sizes=arch['hidden_sizes'],
                dropout_rate=arch['dropout']
            ).to(device)
            self.models.append({'model': model, 'name': arch['name']})
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=50):
        results = []
        
        for i, model_dict in enumerate(self.models):
            print(f"\nTraining ensemble model {i+1}/4: {model_dict['name']}")
            
            model = model_dict['model']
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Setup training
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            best_val_acc = 0
            patience = 15
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for features_batch, labels_batch in train_loader:
                    features_batch = features_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features_batch)
                    loss = loss_fn(outputs, labels_batch)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels_batch.size(0)
                    train_correct += predicted.eq(labels_batch).sum().item()
                
                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for features_batch, labels_batch in val_loader:
                        features_batch = features_batch.to(self.device)
                        labels_batch = labels_batch.to(self.device)
                        outputs = model(features_batch)
                        
                        _, predicted = outputs.max(1)
                        val_total += labels_batch.size(0)
                        val_correct += predicted.eq(labels_batch).sum().item()
                
                val_acc = val_correct / val_total
                scheduler.step()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            results.append(best_val_acc)
            print(f"Model {model_dict['name']} best accuracy: {best_val_acc:.4f}")
        
        return results
    
    def predict_ensemble(self, X):
        """Get ensemble predictions"""
        all_predictions = []
        
        for model_dict in self.models:
            model = model_dict['model']
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                outputs = model(X_tensor)
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
        
        # Average predictions
        ensemble_probs = np.mean(all_predictions, axis=0)
        predictions = np.argmax(ensemble_probs, axis=1)
        
        return predictions, ensemble_probs

class AdvancedMatchPredictor(nn.Module):
    """Enhanced neural network with residual connections"""
    
    def __init__(self, input_size, hidden_sizes=[128, 96, 64, 32], output_size=3, dropout_rate=0.3):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        
        # Build layers with residual connections where possible
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.final_dropout = nn.Dropout(dropout_rate / 2)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input layer
        x = F.relu(self.input_bn(self.input_layer(x)))
        
        # Hidden layers with residual connections
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            identity = x
            x = F.relu(bn(layer(x)))
            x = dropout(x)
            
            # Add residual connection if dimensions match
            if identity.shape[1] == x.shape[1]:
                x = x + identity
        
        # Output layer
        x = self.final_dropout(x)
        x = self.output_layer(x)
        
        return x

def create_meta_features(features_df):
    """Create additional meta-features based on domain knowledge"""
    enhanced = features_df.copy()
    
    # Home advantage indicators
    if 'home_win_rate' in enhanced.columns:
        enhanced['home_advantage_strength'] = (
            enhanced['home_win_rate'] - enhanced.get('away_win_rate', 0.33)
        )
    
    # Team consistency (lower std in ratings = more consistent)
    rating_cols = [col for col in enhanced.columns if 'overall_rating' in col]
    if len(rating_cols) >= 4:
        team1_ratings = [col for col in rating_cols if 'team1' in col]
        team2_ratings = [col for col in rating_cols if 'team2' in col]
        
        if len(team1_ratings) >= 2:
            enhanced['team1_consistency'] = enhanced[team1_ratings].std(axis=1)
            enhanced['team2_consistency'] = enhanced[team2_ratings].std(axis=1)
            enhanced['consistency_advantage'] = (
                enhanced['team2_consistency'] - enhanced['team1_consistency']
            )
    
    # Squad depth (market value distribution)
    market_cols = [col for col in enhanced.columns if 'market_value' in col]
    if len(market_cols) >= 4:
        team1_values = [col for col in market_cols if 'team1' in col]
        team2_values = [col for col in market_cols if 'team2' in col]
        
        if len(team1_values) >= 2:
            enhanced['team1_depth'] = enhanced[team1_values].std(axis=1)
            enhanced['team2_depth'] = enhanced[team2_values].std(axis=1)
            enhanced['depth_advantage'] = enhanced['team1_depth'] - enhanced['team2_depth']
    
    return enhanced

def hybrid_model_prediction(features_path, labels_path, n_folds=5):
    """Combine neural networks with traditional ML models"""
    
    # Load and prepare data
    features = pd.read_csv(features_path)
    
    if 'competition_id' in features.columns:
        features = features.drop('competition_id', axis=1)
    
    # Enhanced preprocessing
    for column in features.columns:
        if features[column].dtype in ['float64', 'int64']:
            if 'rating' in column or 'potential' in column:
                features[column] = features[column].fillna(features[column].median())
            elif 'market_value' in column:
                features[column] = features[column].fillna(0)
                features[column] = features[column].apply(lambda x: np.log1p(x) if x > 0 else 0)
            else:
                features[column] = features[column].fillna(features[column].mean())
    
    # Create meta-features
    features = create_meta_features(features)
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    print(f"Features shape: {features.shape}")
    print(f"Labels: {len(labels)}")
    
    # Prepare data
    features = features.fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Cross-validation with hybrid approach
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    neural_results = []
    ensemble_results = []
    hybrid_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_features, labels)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        X_train, X_val = scaled_features[train_idx], scaled_features[val_idx]
        y_train, y_val = np.array(labels)[train_idx], np.array(labels)[val_idx]
        
        # 1. Neural Network Ensemble
        ensemble = EnsembleMatchPredictor(scaled_features.shape[1], device)
        nn_results = ensemble.train_ensemble(X_train, y_train, X_val, y_val, epochs=50)
        
        nn_predictions, nn_probs = ensemble.predict_ensemble(X_val)
        nn_accuracy = accuracy_score(y_val, nn_predictions)
        neural_results.append(nn_accuracy)
        
        # 2. Traditional ML Models
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        rf_probs = rf.predict_proba(X_val)
        gb_probs = gb.predict_proba(X_val)
        lr_probs = lr.predict_proba(X_val)
        
        # 3. Ensemble of all models (Neural + Traditional)
        all_probs = np.array([nn_probs, rf_probs, gb_probs, lr_probs])
        
        # Weighted ensemble (Neural networks get higher weight)
        weights = np.array([0.4, 0.2, 0.2, 0.2])  # NN gets 40% weight
        final_probs = np.average(all_probs, axis=0, weights=weights)
        final_predictions = np.argmax(final_probs, axis=1)
        
        ensemble_accuracy = accuracy_score(y_val, final_predictions)
        ensemble_results.append(ensemble_accuracy)
        
        # 4. Simple ensemble (equal weights)
        simple_probs = np.mean(all_probs, axis=0)
        simple_predictions = np.argmax(simple_probs, axis=1)
        hybrid_accuracy = accuracy_score(y_val, simple_predictions)
        hybrid_results.append(hybrid_accuracy)
        
        print(f"Neural Network: {nn_accuracy:.4f}")
        print(f"Weighted Ensemble: {ensemble_accuracy:.4f}")
        print(f"Simple Ensemble: {hybrid_accuracy:.4f}")
    
    # Final results
    print(f"\n=== Final Results ===")
    print(f"Neural Network: {np.mean(neural_results):.4f} ± {np.std(neural_results):.4f}")
    print(f"Weighted Ensemble: {np.mean(ensemble_results):.4f} ± {np.std(ensemble_results):.4f}")
    print(f"Simple Ensemble: {np.mean(hybrid_results):.4f} ± {np.std(hybrid_results):.4f}")
    
    # Find best approach
    best_method = "Neural Network" if np.mean(neural_results) >= max(np.mean(ensemble_results), np.mean(hybrid_results)) else \
                  "Weighted Ensemble" if np.mean(ensemble_results) >= np.mean(hybrid_results) else \
                  "Simple Ensemble"
    
    print(f"Best method: {best_method}")
    
    return {
        'neural': (np.mean(neural_results), np.std(neural_results)),
        'weighted': (np.mean(ensemble_results), np.std(ensemble_results)),
        'simple': (np.mean(hybrid_results), np.std(hybrid_results)),
        'best': best_method
    }

# Feature importance analysis
def analyze_feature_importance(features_path, labels_path):
    """Analyze which features are most important"""
    
    features = pd.read_csv(features_path)
    if 'competition_id' in features.columns:
        features = features.drop('competition_id', axis=1)
    
    # Basic preprocessing
    features = features.fillna(features.mean())
    
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    # Use Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    return importance_df