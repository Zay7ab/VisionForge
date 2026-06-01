"""
VisionForge - Backend API v4 (Hugging Face Ready)
MobileNetV3-based transfer learning with analytics persistence

This backend provides a comprehensive REST API for image classification using 
transfer learning. It supports model training, inference, persistence, and analytics.
Built with FastAPI for high performance and easy deployment to Hugging Face Spaces.

Author: VisionForge Team
Version: 4.0
"""

# ============================================================================
# IMPORTS & INITIALIZATION
# ============================================================================
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image, ImageEnhance
import io, os, json, pickle, csv, tempfile
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Initialize FastAPI application with CORS enabled for frontend communication
app = FastAPI(title="VisionForge API v4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================================
# CONFIGURATION
# ============================================================================
# Directory for saving/loading trained model checkpoints
# Supports Hugging Face Spaces environment variable
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Auto-detect and use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# FEATURE EXTRACTION & MODEL ARCHITECTURE
# ============================================================================

class FeatureExtractor:
    """
    Extracts image features using pre-trained MobileNetV3 model.
    
    MobileNetV3 is a lightweight CNN designed for mobile/edge devices.
    We use the ImageNet pre-trained weights and remove the classification head
    to extract rich feature representations from images.
    
    Attributes:
        feature_extractor: Sequential model returning features (576-dim vectors)
        transform: Image preprocessing pipeline (resize, normalize)
    """
    def __init__(self):
        # Load pre-trained MobileNetV3-Small with ImageNet weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        base_model = mobilenet_v3_small(weights=weights)
        
        # Remove the final classification layer to get feature representations
        # The output will be 576-dimensional feature vectors
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor.eval()  # Set to evaluation mode (no dropout/batch norm changes)
        self.feature_extractor.to(device)
        
        # Define image preprocessing: resize to 224x224, normalize using ImageNet stats
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()  # Disable gradient computation for inference
    def extract(self, image_bytes):
        """
        Extract feature vector from image bytes.
        
        Args:
            image_bytes: Raw image data (JPEG/PNG)
            
        Returns:
            numpy array of 576 features
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(device)  # Add batch dimension
        features = self.feature_extractor(tensor)
        return features.squeeze().cpu().numpy()  # Convert to CPU numpy array


class TransferModel(nn.Module):
    """
    Classification head for transfer learning.
    
    Takes pre-extracted features and maps them to class predictions.
    Uses dropout for regularization to prevent overfitting with limited data.
    
    Architecture:
        576 features → 256 (ReLU) → 128 (ReLU) → num_classes logits
    """
    def __init__(self, num_classes, input_features=576):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout: randomly disable 20% of neurons
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output logits (not softmax)
        )
    
    def forward(self, x):
        """
        Forward pass: map features to class logits.
        
        Args:
            x: Batch of feature vectors (B, 576)
            
        Returns:
            Logits for each class (B, num_classes)
        """
        return self.classifier(x)

# Initialize feature extractor globally (loaded once at startup)
feature_extractor = FeatureExtractor()

# ============================================================================
# GLOBAL STATE & DATA STORAGE
# ============================================================================
"""
In-memory storage for current training session.
Stores classes, training images, model state, and prediction history.
In production, this could be replaced with a database.
"""
store = {
    "classes": [],           # List of class names to recognize
    "images": {},            # Dict mapping class_name -> list of {'features', 'raw'} dicts
    "model": None,           # Current trained model (TransferModel instance)
    "encoder": None,         # LabelEncoder for class names -> integers
    "trained": False,        # Whether a model has been trained
    "accuracy": 0.0,         # Best accuracy achieved during training
    "training_history": [],  # List of {'epoch', 'loss', 'accuracy'} dicts
    "prediction_history": [], # List of {'timestamp', 'prediction', 'confidence', 'all'} dicts (last 20)
    "model_settings": {      # User-configurable training hyperparameters
        "augment": True,
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001
    },
    "confusion_matrix": [],  # Confusion matrix from last training
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
def augment_image(image_bytes):
    """
    Generate augmented variants of an image for improved model generalization.
    
    Augmentation techniques:
        - Horizontal flip: mirror the image
        - Brightness: ±30%
        - Contrast: +40% and -40%
        - Rotation: ±10 degrees
        - Color saturation: ±50%
    
    This helps the model learn robust features that aren't sensitive to
    transformations that don't change the object's identity.
    
    Args:
        image_bytes: Raw image data
        
    Returns:
        List of 10 feature vectors (1 original + 9 augmented)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    variants = [img]  # Original image
    
    # Augmentation 1: Horizontal flip
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Augmentation 2-3: Brightness variations
    for factor in [0.7, 1.3]:
        variants.append(ImageEnhance.Brightness(img).enhance(factor))
    
    # Augmentation 4-5: Contrast variations
    variants.append(ImageEnhance.Contrast(img).enhance(1.4))
    variants.append(ImageEnhance.Contrast(img).enhance(0.6))
    
    # Augmentation 6-7: Rotations
    for angle in [-10, 10]:
        variants.append(img.rotate(angle, expand=False))
    
    # Augmentation 8-9: Color saturation
    variants.append(ImageEnhance.Color(img).enhance(1.5))
    variants.append(ImageEnhance.Color(img).enhance(0.5))
    
    # Extract features for each variant
    features = []
    for v in variants:
        buf = io.BytesIO()
        v.save(buf, format="JPEG", quality=90)
        features.append(feature_extractor.extract(buf.getvalue()))
    return features

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """
    Health check endpoint.
    
    Returns:
        API status, version, and model type information.
    """
    return {"status": "running", "version": "4.0", "model": "MobileNetV3-Small"}

@app.post("/classes")
def set_classes(payload: dict):
    """
    Initialize training session with class names.
    
    Resets all previous training data and prepares storage for new classes.
    Requires at least 2 classes to ensure meaningful multi-class classification.
    
    Args:
        payload: {"classes": ["class1", "class2", ...]}
        
    Returns:
        Confirmed list of classes
    """
    names = payload.get("classes", [])
    if len(names) < 2:
        raise HTTPException(400, "Need at least 2 classes")
    
    # Reset store for new training session
    store.update({
        "classes": names, 
        "images": {n: [] for n in names},
        "trained": False, 
        "model": None,
        "training_history": [], 
        "prediction_history": [], 
        "confusion_matrix": [],
    })
    return {"classes": names}

@app.get("/status")
def get_status():
    """
    Get current training state and model information.
    
    Returns:
        Training status, class names, image counts, accuracy, 
        training history, and available device (CPU/GPU)
    """
    return {
        "trained": store["trained"], 
        "classes": store["classes"],
        "counts": {k: len(v) for k, v in store["images"].items()},
        "accuracy": store["accuracy"],
        "training_history": store["training_history"],
        "confusion_matrix": store["confusion_matrix"],
        "model_settings": store["model_settings"],
        "device": str(device)
    }

@app.post("/settings")
def update_settings(payload: dict):
    """
    Update model training hyperparameters.
    
    Args:
        payload: Optional keys:
            - augment (bool): Enable data augmentation
            - epochs (int): Training iterations
            - batch_size (int): Batch size for training
            - learning_rate (float): Optimizer learning rate
            
    Returns:
        Updated settings dictionary
    """
    s = store["model_settings"]
    if "augment" in payload: s["augment"] = bool(payload["augment"])
    if "epochs" in payload: s["epochs"] = int(payload["epochs"])
    if "batch_size" in payload: s["batch_size"] = int(payload["batch_size"])
    if "learning_rate" in payload: s["learning_rate"] = float(payload["learning_rate"])
    return {"settings": s}

@app.post("/upload/{class_name}")
async def upload(class_name: str, file: UploadFile = File(...)):
    """
    Upload a training image for a specific class.
    
    Automatically extracts features using MobileNetV3 and stores them.
    Frontend handles image validation; backend focuses on feature extraction.
    
    Args:
        class_name: Target class name
        file: Image file upload
        
    Returns:
        Updated image count for the class
    """
    if class_name not in store["images"]:
        raise HTTPException(404, f"Class '{class_name}' not found")
    
    data = await file.read()
    try:
        features = feature_extractor.extract(data)
        store["images"][class_name].append({"features": features, "raw": data})
        return {"class": class_name, "count": len(store["images"][class_name])}
    except Exception as e:
        raise HTTPException(400, f"Error processing image: {str(e)}")

@app.post("/train")
def train():
    """
    Train the classification model on uploaded images.
    
    Workflow:
        1. Validate each class has ≥2 images
        2. Optionally augment data to increase training set
        3. Prepare feature matrix and labels
        4. Initialize and train transfer learning model
        5. Evaluate and save best checkpoint
        6. Compute confusion matrix for analysis
    
    Training uses:
        - Adam optimizer with configurable learning rate
        - Learning rate scheduler (reduce on plateau)
        - Early stopping with patience counter
        - Tracks training history (loss & accuracy per epoch)
    
    Returns:
        Training results including final accuracy, number of samples,
        training history, and confusion matrix
    """
    # Validate minimum images per class
    for cls in store["classes"]:
        if len(store["images"].get(cls, [])) < 2:
            raise HTTPException(400, f"Class '{cls}' needs at least 2 images")

    # Get training settings
    settings = store["model_settings"]
    do_augment = settings.get("augment", True)
    epochs = settings.get("epochs", 50)
    learning_rate = settings.get("learning_rate", 0.001)

    # Prepare training data: collect features and labels
    features_list, labels = [], []
    for cls, items in store["images"].items():
        for item in items:
            # Add original image features
            features_list.append(item["features"])
            labels.append(cls)
            
            # Optionally add augmented variants
            if do_augment:
                try:
                    aug_feats = augment_image(item["raw"])
                    for af in aug_feats[1:]:  # Skip original (first one)
                        features_list.append(af)
                        labels.append(cls)
                except:
                    pass  # Skip images that fail augmentation

    # Encode class names to integers (required for PyTorch)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(np.array(labels))
    X = np.array(features_list, dtype=np.float32)
    
    # Convert to PyTorch tensors on device
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y_enc).to(device)
    
    # Initialize model
    model = TransferModel(num_classes=len(store["classes"])).to(device)
    criterion = nn.CrossEntropyLoss()  # Standard cross-entropy for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    history = []
    best_accuracy = 0.0
    patience_counter = 0
    max_patience = 10  # Early stop if no improvement for 10 epochs
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Evaluation phase (on same data for progress tracking)
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).float().mean().item()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Record history
        history.append({
            "epoch": epoch + 1,
            "loss": round(float(loss.item()), 4),
            "accuracy": round(accuracy, 4)
        })
        
        # Early stopping: save best model
        if accuracy > best_accuracy + 0.001:
            best_accuracy = accuracy
            patience_counter = 0
            store["model"] = model.state_dict().copy()
            store["accuracy"] = round(accuracy, 4)
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            break
    
    # Load best model state
    model.load_state_dict(store["model"])
    
    # Compute confusion matrix
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, y_pred = torch.max(outputs, 1)
    cm = confusion_matrix(y_enc, y_pred.cpu().numpy()).tolist()
    
    # Save final state
    store.update({
        "model": model,
        "encoder": encoder,
        "trained": True,
        "training_history": history,
        "confusion_matrix": cm,
    })
    
    return {
        "status": "trained",
        "accuracy": round(best_accuracy * 100, 2),
        "total_samples_with_aug": len(X),
        "classes": list(encoder.classes_),
        "history": history,
        "confusion_matrix": cm,
        "augmented": do_augment,
        "algorithm": "MobileNetV3 Transfer Learning",
        "epochs_trained": len(history)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict class for a given image.
    
    Extracts features from the image and uses the trained model to produce
    class probabilities. Maintains prediction history (last 20 predictions)
    for analytics.
    
    Args:
        file: Image file to classify
        
    Returns:
        Top prediction, confidence percentage, all class probabilities,
        and updated prediction history
    """
    if not store["trained"]:
        raise HTTPException(400, "Model not trained yet")
    
    data = await file.read()
    features = feature_extractor.extract(data)
    X_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Get predictions
    store["model"].eval()
    with torch.no_grad():
        outputs = store["model"](X_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]  # Apply softmax for probabilities
    
    # Format results: sort by probability (descending)
    classes = list(store["encoder"].classes_)
    results = sorted(
        [{"class": c, "probability": round(float(p), 4)} 
         for c, p in zip(classes, probabilities.cpu().numpy())],
        key=lambda x: x["probability"], reverse=True
    )
    
    # Record in prediction history (keep last 20)
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "prediction": results[0]["class"],
        "confidence": round(results[0]["probability"] * 100, 1),
        "all": results,
    }
    store["prediction_history"].insert(0, entry)
    store["prediction_history"] = store["prediction_history"][:20]
    
    return {
        "prediction": results[0]["class"],
        "confidence": round(results[0]["probability"] * 100, 1),
        "probabilities": results,
        "history": store["prediction_history"],
    }

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

@app.post("/model/save")
def save_model(payload: dict):
    """
    Save trained model to disk with metadata.
    
    Stores:
        - Model state dict
        - Label encoder
        - Class names
        - Accuracy
        - Training history
        - Confusion matrix
        - Save timestamp
    
    Args:
        payload: {"name": "model_name"}
        
    Returns:
        Save confirmation
    """
    name = payload.get("name", "model").replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    data = {
        "model_state": store["model"].state_dict() if store["model"] else None,
        "encoder": store["encoder"],
        "classes": store["classes"],
        "accuracy": store["accuracy"],
        "saved_at": datetime.now().isoformat(),
        "model_settings": store["model_settings"],
        "training_history": store["training_history"],
        "confusion_matrix": store["confusion_matrix"],
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return {"status": "saved", "name": name}

@app.get("/model/list")
def list_models():
    """
    List all saved models with their metadata.
    
    Returns:
        List of models with: name, classes, accuracy, save timestamp
    """
    models = []
    if os.path.exists(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.endswith(".pkl"):
                try:
                    with open(os.path.join(MODELS_DIR, fname), "rb") as f:
                        d = pickle.load(f)
                    models.append({
                        "name": fname.replace(".pkl", ""),
                        "classes": d.get("classes", []),
                        "accuracy": d.get("accuracy", 0),
                        "saved_at": d.get("saved_at", ""),
                    })
                except:
                    pass
    return {"models": models}

@app.post("/model/load/{name}")
def load_model(name: str):
    """
    Load a previously saved model into memory.
    
    Reconstructs the model with saved state and prepares for inference
    or continued training.
    
    Args:
        name: Model name (without .pkl extension)
        
    Returns:
        Loaded model info: name, classes, accuracy, history, confusion matrix
    """
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise HTTPException(404, "Model not found")
    with open(path, "rb") as f:
        d = pickle.load(f)
    
    # Reconstruct model
    model = TransferModel(num_classes=len(d["classes"])).to(device)
    model.load_state_dict(d["model_state"])
    
    # Update store
    store.update({
        "model": model,
        "encoder": d["encoder"],
        "classes": d["classes"],
        "images": {c: [] for c in d["classes"]},
        "accuracy": d.get("accuracy", 0),
        "trained": True,
        "model_settings": d.get("model_settings", store["model_settings"]),
        "training_history": d.get("training_history", []),
        "confusion_matrix": d.get("confusion_matrix", []),
    })
    
    return {
        "status": "loaded", 
        "name": name, 
        "classes": store["classes"], 
        "accuracy": store["accuracy"],
        "history": d.get("training_history", []),
        "confusion_matrix": d.get("confusion_matrix", []),
    }

@app.delete("/model/{name}")
def delete_model(name: str):
    """
    Delete a saved model from disk.
    
    Args:
        name: Model name to delete
        
    Returns:
        Deletion confirmation
    """
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if os.path.exists(path):
        os.remove(path)
    return {"status": "deleted"}


# ============================================================================
# ANALYTICS & EXPORT
# ============================================================================

@app.get("/export/csv")
def export_csv():
    """
    Export prediction history as CSV file.
    
    Useful for external analysis and record-keeping.
    
    Returns:
        CSV file with timestamp, prediction, confidence for each prediction
    """
    if not store["prediction_history"]:
        raise HTTPException(400, "No predictions yet")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    writer = csv.writer(tmp)
    writer.writerow(["Timestamp", "Prediction", "Confidence (%)"])
    for entry in store["prediction_history"]:
        writer.writerow([entry["timestamp"], entry["prediction"], entry["confidence"]])
    tmp.close()
    return FileResponse(tmp.name, filename="predictions.csv", media_type="text/csv")

@app.get("/export/report")
def export_report():
    """
    Export comprehensive training report as JSON.
    
    Includes all training metadata, statistics, and prediction history.
    
    Returns:
        JSON report with model info, history, and predictions
    """
    report = {
        "exported_at": datetime.now().isoformat(),
        "classes": store["classes"],
        "accuracy": store["accuracy"],
        "model_settings": store["model_settings"],
        "image_counts": {k: len(v) for k, v in store["images"].items()},
        "training_history": store["training_history"],
        "confusion_matrix": store["confusion_matrix"],
        "prediction_history": store["prediction_history"],
    }
    return report

@app.get("/prediction-history")
def get_prediction_history():
    """
    Retrieve all predictions made so far (last 20).
    
    Returns:
        List of prediction records with timestamps and confidences
    """
    return {"history": store["prediction_history"]}

@app.delete("/prediction-history")
def clear_prediction_history():
    """
    Clear the prediction history.
    
    Returns:
        Confirmation of clearing
    """
    store["prediction_history"] = []
    return {"status": "cleared"}

@app.delete("/reset")
def reset():
    """
    Reset all training data and model state.
    
    Useful for starting a new training session from scratch.
    
    Returns:
        Confirmation of reset
    """
    store.update({
        "classes": [], 
        "images": {}, 
        "model": None,
        "encoder": None, 
        "trained": False, 
        "accuracy": 0.0,
        "training_history": [], 
        "prediction_history": [], 
        "confusion_matrix": [],
    })
    return {"status": "reset"}

# ============================================================================
# SERVER STARTUP
# ============================================================================
# Note: Hugging Face Spaces uses port 7860 by default
# Local deployment can use port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)