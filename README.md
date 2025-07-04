# PADL Project – University of York

This repository contains all code, data, models, and documentation for the Probabilistic & Deep Learning (PADL) 2025 assessment (COM00184M) at the University of York.

## Repository Structure

```
padl-project/
├── README.md               # Project overview and instructions (this file)
├── .gitignore              # Ignore Python caches, models, and env files
├── requirements.txt        # Python dependencies
├── notebooks/              # Jupyter notebooks
│   └── padl.ipynb          # End-to-end PADL assessment
├── src/                    # Python scripts for inference
│   ├── compress_images.py  # Q6: Autoencoder inference
│   ├── predict_waist.py    # Q4: Waist prediction script
│   └── predict_class.py    # Q5: Garment classification script
├── models/                 # Trained weights & scalers
│   ├── scaler_X.pkl        # Q4 input scaler
│   ├── scaler_y.pkl        # Q4 output scaler
│   ├── waist_model.pt      # Q4 trained model
│   ├── encoder.pth         # Q6 encoder weights
│   ├── decoder.pth         # Q6 decoder weights
│   ├── face_autoencoder.pth# Q6 full model weights
│   └── predict_class_weights.pth # Q5 classifier weights
├── data/                   # Raw and preprocessed data files
│   ├── PADL-Q2.csv         # Q2 clustering data
│   ├── PADL-Q3.txt         # Q3 similarity inputs
│   ├── PADL-Q11-train.csv  # Q1 datasets
│   └── body_measurements.csv # Q4 features
├── garment_images/         # Unzipped images for Q5
│   ├── class_0/            # T-shirts
│   ├── class_1/            # Jumpers/Hoodies
│   └── class_2/            # Jeans
├── faces_images/           # Unzipped faces for Q6
│   └── *.png/.jpg
├── results/                # Outputs and logs
│   ├── Q3-result.txt       # Q3 output matrix
│   ├── figures/            # Plots (PCA, clusters, curves)
│   └── logs/               # Training logs
└── docs/                   # Assignment PDF
    └── PADL_M_Assessment_2025-to-publish.pdf
```

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/padl-project.git
   cd padl-project
   ```
2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Notebook

Open `notebooks/padl.ipynb` in Jupyter or Colab and run all cells to reproduce analysis, figures, and results.

### Inference Scripts

- **Q4 Waist Prediction**
  ```bash
  python src/predict_waist.py     --model models/waist_model.pt     --scaler_X models/scaler_X.pkl     --scaler_y models/scaler_y.pkl     --input data/body_measurements.csv     --output results/waist_predictions.csv
  ```

- **Q5 Garment Classification**
  ```bash
  python src/predict_class.py     --weights models/predict_class_weights.pth     --input garment_images/     --output results/classification.csv
  ```

- **Q6 Image Compression**
  ```bash
  python src/compress_images.py     --encoder models/encoder.pth     --decoder models/decoder.pth     --input faces_images/     --output results/compressed_faces/
  ```

## Results & Evaluation

- Check `results/Q3-result.txt` for Q3 similarity matrix.
- Figures in `results/figures/` include PCA scatter and cluster assignments (Q2) and training curves (Q4–Q6).
- Logs in `results/logs/` detail training metrics and SSIM scores.

## Dependencies

```text
numpy
pandas
scikit-learn
matplotlib
gensim
torch
torchvision
scikit-image
jupyterlab
```

## License

This project is submitted in partial fulfilment of COM00184M at the University of York. For academic use only.

## Contact

For questions, open an issue or contact Berke via the VLE discussion board by 5 May 2025.
