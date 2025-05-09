# WeatherML

**WeatherML** is a machine learning project that uses historical weather data to train and evaluate predictive models. This project includes Docker tooling, shell scripts for data handling, and CSV data files used for training and validation. It is designed to run on MacOS, and build .tflite models that are compatible with execution on a Coral Edge TPU

---

## 📦 Project Structure

```
.
├── Dockerfile
├── *.sh
├── train_model.py
├── README.md
├── workspace/*.csv
└── workspace/train_model.py

```
workspace is a shared directory between the container and the host.
---

## 🐳 Docker

The `Dockerfile` sets up a containerized environment with all dependencies needed to run model training and data export scripts.

### Key Features:
- Based on a Python base image with TensorFlow and NumPy
- Installs required Python packages (pandas, scikit-learn, etc.)
- Installs arm64 version of Edge TPU Compiler version 2.0.291256449

### Build & Run:
```bash
docker build -f Dockerfile.tpu -t tpu-dev .
docker run -it --rm -v $(pwd):/workspace tpu-dev bash
```

---

## 🔧 Shell Scripts

The project includes a `.sh` scripts used for automation:

- `run_dev.sh`: Runs the Docker development image once it has been built

Make scripts executable:
```bash
chmod +x *.sh
```

Run them like:
```bash
./run_dev.sh
```

---

## 📁 CSV Data Files

The project uses CSV files as input data. The data is a full year of weather data from my Tempest Weather station:
- `train_data.csv`: Preprocessed weather data for training. Data: April 9th, 2023 - April 8th, 2024
- `val_data.csv`: Preprocessed data for validation. Data: April 9th, 2024 - April 8th, 2025

These files include features such as:
- Temperature
- Humidity
- Pressure
- Timestamps and derived features (e.g., `day_of_year`, `time_of_day`)

Data is clean, numeric, and aligned for supervised learning.

---

## 🧠 Model Training

Run training using:
```bash
python train_model.py
```

The model predicts future temperatures using weather feature history, and the script includes early stopping and learning rate adjustment. Final output is a quantized Edge TPU tflite model.

---


## 📄 License

MIT License or your preferred open-source license.
