lost_found_face_mvp/
│
├── app.py                # Streamlit frontend
├── requirements.txt
│
├── backend/
│   ├── __init__.py
│   ├── detector.py       # face detection (later)
│   ├── embedder.py       # face embeddings (later)
│   ├── matcher.py        # similarity logic (later)
│   └── database.py       # load embeddings & metadata
│
├── data/
│   ├── dataset/          # police face dataset
│   └── labels.json       # child metadata
│
└── utils/
    ├── __init__.py
    └── image_utils.py    # image helpers (later)
