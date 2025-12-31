# Sign Language Recognition Training Pipeline

This project trains Graph Convolutional Network (GCN) and LSTM models for Vietnamese Sign Language recognition using MediaPipe keypoint extraction.

## Prerequisites

- Python 3.11.14
- CUDA 11.8 (for GPU training)
- Google Drive account (for dataset storage)
- Kaggle account (for GPU training)

## Install dependencies:
```bash
pip install -r requirements.txt
```


## Pipeline Steps

### 1. Download Dataset

**Script:** `data/download_data.py`

Crawls sign language videos from qipedc.moet.gov.vn and saves them to the Videos folder.

**Note:** Pre-crawled and curated videos are available at:
- [Download Videos](https://drive.google.com/drive/folders/1Qhx6PEqhFTYbGtkwnM53A5yad-WPO08u?usp=sharing)

Download the Videos folder from this link instead of running the script.

### 2. Extract Keypoints

**Script:** `data/checkpoint_extraction.py`  
**Helper:** `data/augment_function.py`

Extracts pose, face, and hand keypoints using MediaPipe and saves them to the `/Data` folder with data augmentation.

```bash
python data/checkpoint_extraction.py
```

**Features:**
- MediaPipe Holistic keypoint extraction
- Data augmentation (scaling, rotation, noise, etc.)
- Multi-threaded processing

### 3. Upload to Kaggle

After keypoint extraction, zip the `/Data` folder for each topic and upload to Kaggle for GPU training.

**Pre-extracted Data:** Available at the same [Google Drive link](https://drive.google.com/drive/folders/1Qhx6PEqhFTYbGtkwnM53A5yad-WPO08u?usp=sharing)

Each topic has its own `/Data` folder with extracted keypoints.

### 4. Train GCN Models

**Script:** `data/train-gcn.py`

Upload this script to Kaggle and run with GPU to train GCN models. Models are saved as `.pth` files.

**Training output:**
- One `.pth` model file per topic
- Pre-trained models available at [Google Drive](https://drive.google.com/drive/folders/1bAmg1NmsY_PglK5_9G039aH2fO8tyvfk?usp=sharing)

### 5. Train LSTM Models

**Script:** `data/train-lstm.py`

Upload this script to Kaggle and run with GPU to train LSTM models. Models are saved as `.pth` files.

### 6. Deploy Models

Copy trained `.pth` files to:
```
# For GCN
project/training/khoa/app/backend/Model/gcn/

# For LSTM
project/training/khoa/app/backend/Model/lstm/
```

**Model structure:**
- Each topic has its own GCN model
- LSTM models are also supported
- Models are loaded automatically by `GCNPredictionService` and `LSTMPredictionService`

### 7. Test API

**Directory:** `app/backend/`

Start the FastAPI server:
```bash
cd app/backend
uvicorn app:app --reload
```

**API Endpoints:**
- `GET /api/words` - Get test words
- `POST /api/predict-gcn` - Predict sign language (GCN)
- `POST /api/predict-lstm` - Predict sign language (LSTM)
- `GET /api/videos/{video_name}` - Serve video files

**Modify test words:**
Edit `selected_ids` in `app/backend/src/routers/api.py`:
```python
selected_ids = [142, 143, 144, 145, 146, 147]  # Change these IDs
```

**Testing:**
- Go live: `project\training\khoa\app\frontend\index.html`
- Open browser: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Project Structure

```
.
├── app/
│   ├── backend/
│   │   ├── app.py                # FastAPI application
│   │   ├── Model/
│   │   │   ├── gcn/              # GCN model files (.pth)
│   │   │   ├── lstm/             # LSTM model files (.pth)
│   │   │   └── ...               # JSON data files
│   │   └── src/
│   │       ├── models/           # Database models
│   │       ├── routers/          # API routes
│   │       ├── schemas/          # Pydantic schemas
│   │       ├── services/         # Business logic
│   │       └── utils/            # Utility functions
│   └── frontend/
│       └── index.html            # Web interface
├── data/
│   ├── augment_function.py       # Data augmentation
│   ├── checkpoint_extraction.py  # Keypoint extraction
│   ├── constants.py              # Data constants
│   ├── download_data.py          # Web crawler
│   ├── train-gcn.py              # GCN training script
│   └── train-lstm.py             # LSTM training script
├── requirements.txt
└── test_visualize.py
```

## Topics

The system supports 10 sign language topics:
- Alphabet (chu_cai)
- Numbers (so_dem)
- Family (gia_dinh)
- Weather (thoi_tiet)
- Actions (hanh_dong)
- Objects (do_vat)
- Common words (tu_thong_dung)
- Questions (cau_hoi)
- Professions (nghe_nghiep)
- Personality (tinh_cach)

## Notes

- Use GPU for training on Kaggle (free tier available)
- Each topic requires separate model training
- Pre-trained models and datasets save significant time
- All resources available at [Google Drive](https://drive.google.com/drive/folders/1Qhx6PEqhFTYbGtkwnM53A5yad-WPO08u?usp=sharing)

## Troubleshooting

**Import errors:** Ensure all dependencies from `requirements.txt` are installed  
**CUDA errors:** Verify CUDA 11.8 compatibility with your GPU  
**API connection:** Check if port 8000 is available
