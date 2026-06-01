# 🧠 VisionForge - AI Image Classifier

A powerful, no-code AI image classification platform that allows anyone to train custom deep learning models using transfer learning. Built with FastAPI, PyTorch, and Streamlit.

![Version](https://img.shields.io/badge/version-4.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)

---

## 🚀 Features

- **Transfer Learning with MobileNetV3**: Pre-trained model for efficient feature extraction
- **No Coding Required**: Intuitive web interface for training and predictions
- **Data Augmentation**: Automatic image augmentation (flips, rotations, brightness adjustments) to improve model robustness
- **Real-Time Predictions**: Instant image classification with confidence scores
- **Advanced Analytics**: Training curves, confusion matrices, and detailed metrics
- **Model Persistence**: Save, load, and manage multiple trained models
- **GPU Support**: Automatic detection and utilization of CUDA devices
- **100% Private**: All processing done locally, data never leaves your machine
- **Dockerized**: Easy deployment with Docker and Docker Compose

---

## 📋 Prerequisites

- **Docker** and **Docker Compose** installed
  - [Download Docker Desktop](https://docs.docker.com/get-docker/)
- **Python 3.8+** (for local development)
- **4GB RAM minimum** (8GB recommended for smooth training)

---

## ⚡ Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VisionForge.git
   cd VisionForge
   ```

2. **Run the application**
   ```bash
   python run.py
   ```
   
   This will:
   - Build Docker images
   - Start the backend and frontend services
   - Open your browser to http://localhost:8501

3. **Stop the application**
   ```bash
   docker compose down
   ```

### Option 2: Local Development

1. **Install dependencies**
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   pip install -r requirements.txt
   ```

2. **Start backend**
   ```bash
   cd backend
   python main.py
   # Backend runs on http://localhost:8000
   ```

3. **Start frontend (in new terminal)**
   ```bash
   cd frontend
   streamlit run app.py
   # Frontend runs on http://localhost:8501
   ```

---

## 🎯 Usage Guide

### 1. Define Classes
- Enter the types of objects you want to recognize (e.g., "cat", "dog", "bird")
- Minimum 2 classes required

### 2. Upload Training Images
- Upload at least 3-5 images per class
- Supported formats: JPEG, PNG, WebP, BMP
- The model will automatically augment images to improve learning

### 3. Train Model
- Click "Start Training"
- Model trains using transfer learning (takes 1-3 minutes)
- View real-time accuracy and loss curves

### 4. Make Predictions
- Upload an image to classify
- Get instant prediction with confidence score
- View probabilities for all classes

### 5. Analyze Results
- View training history
- Inspect confusion matrix
- Export predictions as CSV

### 6. Save & Export
- Save trained models for later use
- Export predictions and reports
- Load previously saved models

---

## 🏗️ Architecture

### Backend (FastAPI)
```
backend/
├── main.py              # FastAPI server with ML endpoints
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── saved_models/        # Model checkpoints storage
```

**Key Components:**
- `FeatureExtractor`: Pre-trained MobileNetV3 for feature extraction
- `TransferModel`: Lightweight classifier on top of features
- Model Training: Adam optimizer with learning rate scheduling
- Analytics: Confusion matrix, training history

**API Endpoints:**
- `POST /classes` - Initialize training session
- `POST /upload/{class_name}` - Upload training images
- `POST /train` - Train the model
- `POST /predict` - Classify an image
- `POST /model/save` - Save trained model
- `GET /model/list` - List saved models
- `POST /model/load/{name}` - Load a saved model

### Frontend (Streamlit)
```
frontend/
├── app.py               # Streamlit UI application
├── Dockerfile           # Container configuration
└── requirements.txt     # Python dependencies
```

**Pages:**
- 🏠 **Home**: Overview and features
- 🎓 **Train**: Define classes, upload images, train model
- 🔍 **Predict**: Make predictions with trained model
- 📊 **Analytics**: View training curves and confusion matrix
- 💾 **Models**: Manage saved models
- ⚙️ **Settings**: Configure training hyperparameters

### Data Flow
```
User Image
    ↓
Streamlit Frontend
    ↓
FastAPI Backend
    ↓
Feature Extraction (MobileNetV3)
    ↓
Classification Head
    ↓
Prediction + Confidence
    ↓
Display in Frontend
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework
- **TorchVision**: Pre-trained models and image utilities
- **scikit-learn**: ML utilities (LabelEncoder, confusion matrix)
- **Pillow**: Image processing

### Frontend
- **Streamlit**: Rapid web app development
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Requests**: HTTP client for API calls

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Uvicorn**: ASGI server for FastAPI

---

## 📊 Model Details

### Transfer Learning Approach
- **Pre-trained Model**: MobileNetV3-Small (ImageNet weights)
- **Feature Extraction**: 576-dimensional feature vectors
- **Classification Head**:
  ```
  576 features
    ↓
  Dense(256) + ReLU
    ↓
  Dropout(0.2)
    ↓
  Dense(128) + ReLU
    ↓
  Dense(num_classes)
  ```

### Training Strategy
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: 10 epochs without improvement
- **Data Augmentation**: 9x multiplication (original + 8 variants)

### Why MobileNetV3?
- Lightweight and fast (ideal for cloud deployment)
- High accuracy despite small size
- Excellent transfer learning foundation
- Efficient on CPU and GPU

---

## 📈 Performance

### Typical Results
- **Training Time**: 1-3 minutes (with 50-100 images)
- **Prediction Speed**: <100ms per image
- **Accuracy**: 85-99% (depends on data quality and complexity)
- **Memory Usage**: ~1.5GB (with GPU), ~2.5GB (without GPU)

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB |
| GPU | None | NVIDIA with 2GB+ VRAM |
| Storage | 2 GB | 5 GB |

---

## 🐛 Troubleshooting

### Docker Issues
```bash
# Check if Docker is running
docker version

# View container logs
docker compose logs -f

# Rebuild images
docker compose up --build

# Clean up everything
docker compose down -v
```

### Backend Connection Issues
- Verify backend is running: `curl http://localhost:8000/`
- Check API docs: http://localhost:8000/docs
- View backend logs: `docker compose logs backend -f`

### Out of Memory
- Reduce batch size in Settings
- Disable data augmentation
- Use GPU if available
- Restart containers: `docker compose restart`

### Model Training Issues
- Ensure each class has at least 2 images
- Check image formats are supported
- Try enabling/disabling data augmentation
- View backend logs for detailed errors

---

## 📦 Deployment

### Hugging Face Spaces
The backend is configured for Hugging Face Spaces deployment:
- Update `API` URL in `frontend/app.py`
- Backend runs on port 7860 (HF default)

### Cloud Deployment (AWS, GCP, Azure)
1. Push Docker images to container registry
2. Update `docker-compose.yml` with cloud endpoints
3. Deploy using cloud orchestration tools (ECS, GKE, ACI)

### Local Network Access
```bash
# Find your local IP
ipconfig getifaddr en0  # macOS
hostname -I              # Linux

# Access from other devices
http://YOUR_LOCAL_IP:8501
```

---

## 🔐 Security

- **No Data Upload**: All processing is local
- **Model Privacy**: Models stored locally, not sent anywhere
- **CORS Enabled**: Frontend and backend can run on same/different servers
- **No Authentication**: Suitable for private/internal use

For production use, consider adding:
- Authentication layer
- HTTPS/TLS encryption
- Rate limiting
- Input validation

---

## 📝 File Structure

```
VisionForge/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── Dockerfile           # Backend container config
│   ├── requirements.txt     # Python dependencies
│   └── saved_models/        # Trained model storage
├── frontend/
│   ├── app.py               # Streamlit application
│   ├── Dockerfile           # Frontend container config
│   └── requirements.txt     # Python dependencies
├── docker-compose.yml       # Multi-container orchestration
├── run.py                   # Docker launcher script
├── install.bat              # Windows installation script
├── README.md                # This file
└── .gitignore               # Git ignore rules
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Multi-GPU training support
- Model quantization for mobile deployment
- Additional augmentation techniques
- Advanced hyperparameter tuning UI
- Batch prediction mode

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **MobileNetV3**: TorchVision pre-trained models
- **Streamlit**: Rapid web app framework
- **FastAPI**: Modern Python web framework
- **PyTorch**: Deep learning framework

---

## 📧 Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review API documentation: http://localhost:8000/docs
3. Check backend logs: `docker compose logs backend -f`

---

## 🚀 Roadmap

- [ ] Multi-GPU training support
- [ ] Model export (ONNX, TorchScript)
- [ ] Mobile app integration
- [ ] Real-time webcam training
- [ ] Advanced augmentation strategies
- [ ] Federated learning support
- [ ] Multi-task learning
- [ ] Model compression for edge devices

---

**Made with ❤️ by the VisionForge Team**

**Version 4.0** | Built with PyTorch & FastAPI & Streamlit
