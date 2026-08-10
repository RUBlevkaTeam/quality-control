import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class ProductQualityPredictor:
    def __init__(self):
        # Dictionary to hold models and thresholds
        self.category_models = {}

    def train(self, train_df):
        self.category_models = {}

        for category in train_df['category'].unique():
            print(f"--- Training model for category: {category} ---")
            
            cat_train = train_df[train_df['category'] == category]
            X = np.vstack(cat_train['embedding'].values)
            y = cat_train['label'].values
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_train, y_train)
            
            # Tune threshold
            val_probs = clf.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1, best_thresh = 0, 0.5
            
            for t in thresholds:
                val_preds = (val_probs >= t).astype(int)
                current_f1 = f1_score(y_val, val_preds)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_thresh = t
                    
            print(f"Optimal Threshold: {best_thresh:.2f} (Val F1: {best_f1:.4f})")
            
            # Retrain on all category data
            clf_final = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf_final.fit(X, y)
            
            self.category_models[category] = {
                'model': clf_final,
                'threshold': best_thresh
            }

    def predict(self, embeddings, categories):
        probs = []
        preds = []
        
        for emb, cat in zip(embeddings, categories):
            if cat not in self.category_models:
                # Fallback in case of unseen categories
                probs.append(0.0)
                preds.append(0)
                continue
                
            model_info = self.category_models[cat]
            clf = model_info['model']
            thresh = model_info['threshold']
            
            # predict_proba expects 2D array
            emb_2d = np.array(emb).reshape(1, -1)
            prob = clf.predict_proba(emb_2d)[0][1]
            pred = int(prob >= thresh)
            
            probs.append(prob)
            preds.append(pred)
            
        return probs, preds

    def save(self, filepath):
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)
    
