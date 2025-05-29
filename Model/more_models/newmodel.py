import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ImprovedMatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=3, dropout_rate=0.2):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.out = nn.Linear(hidden_sizes[2], output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.out(x)
        return x

def create_minimal_features(features_df):
    """Create only the most important differential features"""
    enhanced_features = features_df.copy()
    
    # Only create key differential features that matter for football
    
    # 1. Overall team strength difference (most important)
    team1_strength = (
        enhanced_features['team1_defender_overall_rating'] * 0.2 +
        enhanced_features['team1_goalkeeper_overall_rating'] * 0.15 +
        enhanced_features['team1_attack_overall_rating'] * 0.35 +
        enhanced_features['team1_midfield_overall_rating'] * 0.3
    )
    
    team2_strength = (
        enhanced_features['team2_defender_overall_rating'] * 0.2 +
        enhanced_features['team2_goalkeeper_overall_rating'] * 0.15 +
        enhanced_features['team2_attack_overall_rating'] * 0.35 +
        enhanced_features['team2_midfield_overall_rating'] * 0.3
    )
    
    enhanced_features['strength_difference'] = team1_strength - team2_strength
    
    # 2. Attack vs Defense matchup (crucial for football)
    enhanced_features['attack_defense_advantage'] = (
        enhanced_features['team1_attack_overall_rating'] - enhanced_features['team2_defender_overall_rating']
    )
    enhanced_features['defense_attack_advantage'] = (
        enhanced_features['team2_attack_overall_rating'] - enhanced_features['team1_defender_overall_rating']
    )
    
    # 3. Market value difference (indicates squad quality)
    team1_market_cols = [col for col in enhanced_features.columns if 'team1_' in col and 'market_value' in col]
    team2_market_cols = [col for col in enhanced_features.columns if 'team2_' in col and 'market_value' in col]
    
    if team1_market_cols and team2_market_cols:
        team1_total_value = enhanced_features[team1_market_cols].sum(axis=1)
        team2_total_value = enhanced_features[team2_market_cols].sum(axis=1)
        enhanced_features['market_value_difference'] = team1_total_value - team2_total_value
    
    # 4. Form difference (if available)
    if 'home_form' in enhanced_features.columns and 'away_form' in enhanced_features.columns:
        enhanced_features['form_difference'] = enhanced_features['home_form'] - enhanced_features['away_form']
    
    # 5. Win rate difference (if available)
    if 'home_win_rate' in enhanced_features.columns and 'away_win_rate' in enhanced_features.columns:
        enhanced_features['win_rate_difference'] = enhanced_features['home_win_rate'] - enhanced_features['away_win_rate']
    
    return enhanced_features

def prepare_features_improved(csv_path):
    """Improved but conservative feature preparation"""
    features = pd.read_csv(csv_path)
    
    print(f"Original features shape: {features.shape}")
    
    # Remove competition_id if present
    if 'competition_id' in features.columns:
        features = features.drop('competition_id', axis=1)
    
    # Handle missing values (keep your original approach that worked)
    for column in features.columns:
        if features[column].dtype in ['float64', 'int64']:
            features[column] = features[column].fillna(features[column].mean())
    
    # Handle form columns specifically
    form_cols = ['home_form', 'away_form', 'home_win_rate', 'home_draw_rate', 'away_win_rate', 'away_draw_rate']
    for col in form_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0.5)
    
    # Apply log transform to market values (keep your approach)
    market_value_cols = [col for col in features.columns if 'market_value' in col]
    for col in market_value_cols:
        features[col] = features[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    
    # Add only essential differential features
    features = create_minimal_features(features)
    
    print(f"Enhanced features shape: {features.shape}")
    
    return features

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def train_model_improved(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Gradient clipping (conservative)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), correct / total

def evaluate_model_improved(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), correct / total, all_preds, all_labels

def train_improved_model(config, features_path, labels_path):
    """Conservative improvement on your working model"""
    
    # Load and prepare features
    features = prepare_features_improved(features_path)
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    print(f"Features shape: {features.shape}")
    print(f"Number of labels: {len(labels)}")
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Scale features (same as your approach)
    features = features.fillna(0)
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create datasets with same split as your original
    dataset = TensorDataset(
        torch.tensor(scaled_features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Use same batch size as your original
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model with more conservative settings
    model = ImprovedMatchPredictor(
        input_size=scaled_features.shape[1], 
        hidden_sizes=[64, 32, 16],  # Similar to your original
        dropout_rate=0.1  # Lower dropout
    )
    
    # Setup device (fix your original bug)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Loss function with class weights (same as your approach)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use Adam with conservative settings
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15)
    
    # Training loop
    best_accuracy = config.get("best_accuracy", 0.0)
    
    for epoch in range(config.get("number_of_epochs", 50)):
        train_loss, train_acc = train_model_improved(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate_model_improved(model, val_loader, loss_fn, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), config.get("model_path", "improved_model.pth"))
            print(f"★ New best accuracy: {best_accuracy:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nFinal best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(config.get("model_path", "improved_model.pth")))
    val_loss, val_acc, val_preds, val_labels = evaluate_model_improved(model, val_loader, loss_fn, device)
    
    # Print detailed results
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, 
                              target_names=['Home Win', 'Draw', 'Away Win']))
    
    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Home Win', 'Draw', 'Away Win'],
                yticklabels=['Home Win', 'Draw', 'Away Win'])
    plt.title(f'Confusion Matrix - Accuracy: {val_acc:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return model, scaler, best_accuracy