import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class AdvancedMatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 96, 64, 32], output_size=3, dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate if i < len(hidden_sizes)-1 else dropout_rate/2))
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.feature_layers(x)
        x = self.output_layer(x)
        return x

def create_advanced_features(features_df):
    """Create comprehensive football-specific features"""
    enhanced_features = features_df.copy()
    
    # 1. Overall team strength with position-weighted importance
    def calculate_team_strength(prefix):
        defense_weight = 0.25
        goalkeeper_weight = 0.15
        attack_weight = 0.35
        midfield_weight = 0.25
        
        strength = (
            enhanced_features[f'{prefix}_defender_overall_rating'] * defense_weight +
            enhanced_features[f'{prefix}_goalkeeper_overall_rating'] * goalkeeper_weight +
            enhanced_features[f'{prefix}_attack_overall_rating'] * attack_weight +
            enhanced_features[f'{prefix}_midfield_overall_rating'] * midfield_weight
        )
        return strength
    
    team1_strength = calculate_team_strength('team1')
    team2_strength = calculate_team_strength('team2')
    
    # Overall strength difference
    enhanced_features['strength_difference'] = team1_strength - team2_strength
    enhanced_features['strength_ratio'] = team1_strength / (team2_strength + 1e-6)
    
    # 2. Position-specific matchups
    enhanced_features['attack_vs_defense_1'] = (
        enhanced_features['team1_attack_overall_rating'] - 
        enhanced_features['team2_defender_overall_rating']
    )
    enhanced_features['attack_vs_defense_2'] = (
        enhanced_features['team2_attack_overall_rating'] - 
        enhanced_features['team1_defender_overall_rating']
    )
    enhanced_features['midfield_battle'] = (
        enhanced_features['team1_midfield_overall_rating'] - 
        enhanced_features['team2_midfield_overall_rating']
    )
    enhanced_features['goalkeeper_advantage'] = (
        enhanced_features['team1_goalkeeper_overall_rating'] - 
        enhanced_features['team2_goalkeeper_overall_rating']
    )
    
    # 3. Market value features (squad depth indicator)
    market_cols_1 = [col for col in enhanced_features.columns if 'team1_' in col and 'market_value' in col]
    market_cols_2 = [col for col in enhanced_features.columns if 'team2_' in col and 'market_value' in col]
    
    if market_cols_1 and market_cols_2:
        team1_total_value = enhanced_features[market_cols_1].sum(axis=1)
        team2_total_value = enhanced_features[market_cols_2].sum(axis=1)
        
        enhanced_features['market_value_difference'] = team1_total_value - team2_total_value
        enhanced_features['market_value_ratio'] = team1_total_value / (team2_total_value + 1e6)
        
        # Value per rating (efficiency indicator)
        enhanced_features['team1_value_efficiency'] = team1_total_value / (team1_strength + 1e-6)
        enhanced_features['team2_value_efficiency'] = team2_total_value / (team2_strength + 1e-6)
        enhanced_features['value_efficiency_difference'] = (
            enhanced_features['team1_value_efficiency'] - 
            enhanced_features['team2_value_efficiency']
        )
    
    # 4. Potential vs Current ability (youth factor)
    def calculate_potential_gap(prefix):
        potential_cols = [col for col in enhanced_features.columns 
                         if f'{prefix}_' in col and 'potential' in col]
        rating_cols = [col.replace('potential', 'overall_rating') for col in potential_cols]
        
        if all(col in enhanced_features.columns for col in rating_cols):
            potential_sum = enhanced_features[potential_cols].sum(axis=1)
            rating_sum = enhanced_features[rating_cols].sum(axis=1)
            return potential_sum - rating_sum
        return 0
    
    enhanced_features['team1_potential_gap'] = calculate_potential_gap('team1')
    enhanced_features['team2_potential_gap'] = calculate_potential_gap('team2')
    enhanced_features['potential_gap_difference'] = (
        enhanced_features['team1_potential_gap'] - enhanced_features['team2_potential_gap']
    )
    
    # 5. Form and historical performance
    form_cols = ['home_form', 'away_form', 'home_win_rate', 'home_draw_rate', 
                 'away_win_rate', 'away_draw_rate']
    
    for col in form_cols:
        if col in enhanced_features.columns:
            enhanced_features[col] = enhanced_features[col].fillna(0.33)  # Neutral assumption
    
    if 'home_form' in enhanced_features.columns and 'away_form' in enhanced_features.columns:
        enhanced_features['form_difference'] = enhanced_features['home_form'] - enhanced_features['away_form']
        enhanced_features['combined_form'] = (enhanced_features['home_form'] + enhanced_features['away_form']) / 2
    
    if 'home_win_rate' in enhanced_features.columns and 'away_win_rate' in enhanced_features.columns:
        enhanced_features['win_rate_difference'] = enhanced_features['home_win_rate'] - enhanced_features['away_win_rate']
        enhanced_features['home_advantage'] = enhanced_features['home_win_rate'] - enhanced_features['home_draw_rate']
        enhanced_features['away_strength'] = enhanced_features['away_win_rate'] - enhanced_features['away_draw_rate']
    
    # 6. Balance indicators (how well-rounded teams are)
    def calculate_balance(prefix):
        positions = ['defender', 'goalkeeper', 'attack', 'midfield']
        ratings = [enhanced_features[f'{prefix}_{pos}_overall_rating'] for pos in positions]
        
        if all(col.notna().all() for col in ratings):
            ratings_array = np.array(ratings).T
            return np.std(ratings_array, axis=1)  # Lower std = more balanced
        return np.zeros(len(enhanced_features))
    
    enhanced_features['team1_balance'] = calculate_balance('team1')
    enhanced_features['team2_balance'] = calculate_balance('team2')
    enhanced_features['balance_difference'] = enhanced_features['team2_balance'] - enhanced_features['team1_balance']
    
    # 7. Interaction features
    enhanced_features['strength_form_interaction'] = (
        enhanced_features['strength_difference'] * enhanced_features.get('form_difference', 0)
    )
    
    if 'market_value_difference' in enhanced_features.columns:
        enhanced_features['value_strength_correlation'] = (
            enhanced_features['market_value_difference'] * enhanced_features['strength_difference']
        )
    
    return enhanced_features

def prepare_features_advanced(csv_path):
    """Advanced feature preparation with better handling"""
    features = pd.read_csv(csv_path)
    print(f"Original features shape: {features.shape}")
    
    # Remove competition_id
    if 'competition_id' in features.columns:
        features = features.drop('competition_id', axis=1)
    
    # Handle missing values more intelligently
    for column in features.columns:
        if features[column].dtype in ['float64', 'int64']:
            if 'rating' in column or 'potential' in column:
                # For ratings, use median of similar position
                features[column] = features[column].fillna(features[column].median())
            elif 'market_value' in column:
                # For market values, use log transform and handle zeros
                features[column] = features[column].fillna(0)
                features[column] = features[column].apply(lambda x: np.log1p(x) if x > 0 else 0)
            else:
                features[column] = features[column].fillna(features[column].mean())
    
    # Apply log transform to market values
    market_value_cols = [col for col in features.columns if 'market_value' in col]
    for col in market_value_cols:
        if not features[col].apply(lambda x: x == np.log1p(x) if x > 0 else x == 0).all():
            features[col] = features[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    
    # Create advanced features
    features = create_advanced_features(features)
    
    # Remove highly correlated features
    numeric_features = features.select_dtypes(include=[np.number])
    correlation_matrix = numeric_features.corr().abs()
    
    # Find pairs of highly correlated features (threshold: 0.95)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
    
    # Remove one from each highly correlated pair
    to_remove = set()
    for pair in high_corr_pairs:
        if pair[1] not in to_remove:  # Keep the first one, remove the second
            to_remove.add(pair[1])
    
    if to_remove:
        print(f"Removing {len(to_remove)} highly correlated features: {to_remove}")
        features = features.drop(columns=list(to_remove))
    
    print(f"Enhanced features shape: {features.shape}")
    return features

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_with_cross_validation(config, features_path, labels_path, n_folds=5):
    """Train with k-fold cross validation for better evaluation"""
    
    # Load and prepare features
    features = prepare_features_advanced(features_path)
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    print(f"Features shape: {features.shape}")
    print(f"Number of labels: {len(labels)}")
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Scale features
    features = features.fillna(0)
    scaler = StandardScaler()  # Try StandardScaler instead
    scaled_features = scaler.fit_transform(features)
    
    # Cross-validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_features, labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Create datasets for this fold
        X_train, X_val = scaled_features[train_idx], scaled_features[val_idx]
        y_train, y_val = np.array(labels)[train_idx], np.array(labels)[val_idx]
        
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
        
        # Initialize model
        model = AdvancedMatchPredictor(
            input_size=scaled_features.shape[1],
            hidden_sizes=[128, 96, 64, 32],
            dropout_rate=0.3
        ).to(device)
        
        # Loss function with class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        # Try Focal Loss for better handling of imbalanced classes
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Optimizer with different learning rate and weight decay
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training loop for this fold
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features_batch, labels_batch in train_loader:
                features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(features_batch)
                loss = loss_fn(outputs, labels_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features_batch, labels_batch in val_loader:
                    features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
                    outputs = model(features_batch)
                    loss = loss_fn(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels_batch.size(0)
                    val_correct += predicted.eq(labels_batch).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        fold_results.append(best_val_acc)
        print(f"Fold {fold + 1} best accuracy: {best_val_acc:.4f}")
    
    # Final results
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in fold_results]}")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Best fold: {max(fold_results):.4f}")
    
    return mean_acc, std_acc, fold_results
