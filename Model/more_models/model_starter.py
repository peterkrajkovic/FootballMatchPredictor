  
from Model.more_models.advancved_model import train_with_cross_validation
from Model.more_models.ensembleMatchPredictor import analyze_feature_importance, hybrid_model_prediction
from Model.more_models.newmodel import train_improved_model


## odtialto nejde spustit, iba z mainu, bacha na path pre subory
config = {
    "number_of_epochs": 50,
    "model_path": "improved_football_model.pth",
    "best_accuracy": 0.0
}

# Train the enhanced model
model, scaler, best_acc = train_improved_model(
    config, 
    "features_all_cID.csv", 
    "labels_all_cID.txt"
)

######################################################################
## odtialto nejde spustit, iba z mainu

config = {
    "number_of_epochs": 100,
    "model_path": "enhanced_football_model.pth",
    "best_accuracy": 0.0
}

#Assuming your files are named accordingly
features_path =  "features_all_cID.csv"  # Update with your actual path
labels_path = "labels_all_cID.txt"     # Update with your actual path

#Run cross-validation
mean_acc, std_acc, fold_results = train_with_cross_validation(
    config, features_path, labels_path, n_folds=5
)

#############################################################

## odtialto nejde spustit, iba z mainu

results = hybrid_model_prediction("features_all_cID.csv","labels_all_cID.txt" , n_folds=2)
    
importance_df = analyze_feature_importance("features_all_cID.csv", "labels_all_cID.txt" )

###############################################################
