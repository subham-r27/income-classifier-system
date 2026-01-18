# Income Prediction System

A web application for predicting whether an individual's income is greater than or less than $50,000 using machine learning models. The system supports two models: K-Nearest Neig

https://github.com/user-attachments/assets/3e577fd9-1de3-4cc0-a59c-3b5c28f3b81a

hbors (KNN) and Logistic Regression.

## Features

- **Interactive Web Interface**: User-friendly form to input personal and financial information
- **Multiple ML Models**: Choose between KNN or Logistic Regression for predictions
- **Real-time Predictions**: Get instant income predictions based on input data
- **Dynamic Form Options**: Dropdown menus populated with valid values from the dataset

## Project Structure

```
Project/
├── app/
│   ├── main.py              # FastAPI application and API endpoints
│   ├── index.html           # Frontend HTML interface
│   └── static/
│       ├── script.js        # Frontend JavaScript logic
│       └── styles.css       # CSS styling
├── datasets/
│   └── income(1).csv        # Training dataset
├── dev/
│   ├── knn_model.ipynb      # KNN model development notebook
│   ├── log_reg_model.ipynb  # Logistic Regression model development notebook
│   └── models/              # Trained model files
│       ├── knn_model.pkl
│       ├── logistic_regression_model.pkl
│       ├── scaler.pkl
│       └── feature_names.pkl
├── run_app.py               # Application entry point
└── requirements.txt         # Python dependencies
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd Lecture_week4
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the FastAPI server**:
   ```bash
   python run_app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://127.0.0.1:8000
   ```
   or
   ```
   http://localhost:8000
   ```

3. **Use the web interface** to:
   - Select a model (KNN or Logistic Regression)
   - Fill in the form with personal, employment, and financial information
   - Click "Predict Income" to get the prediction

## API Endpoints

### `GET /`
Returns the main HTML page.

### `GET /api/unique-values`
Returns unique values for all categorical fields (JobType, EdType, maritalstatus, etc.) used to populate dropdown menus.

**Response:**
```json
{
  "JobType": ["Federal-gov", "Local-gov", ...],
  "EdType": ["Bachelors", "Masters", ...],
  ...
}
```

### `POST /api/predict`
Makes an income prediction based on the provided input data.

**Request Body:**
```json
{
  "age": 35,
  "JobType": "Private",
  "EdType": "Bachelors",
  "maritalstatus": "Married-civ-spouse",
  "occupation": "Tech-support",
  "relationship": "Husband",
  "race": "White",
  "gender": "Male",
  "capitalgain": 0,
  "capitalloss": 0,
  "hoursperweek": 40,
  "nativecountry": "United-States",
  "model_type": "knn"
}
```

**Response:**
```json
{
  "prediction": 1,
  "result": "greater than 50,000",
  "model_used": "knn"
}
```

## Technologies Used

- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: pandas, numpy

## Model Information

The application uses two pre-trained machine learning models:

1. **K-Nearest Neighbors (KNN)**: A non-parametric classification algorithm
2. **Logistic Regression**: A linear classification algorithm with feature scaling

Both models were trained on the income dataset and saved as pickle files in the `dev/models/` directory.

## Development

Model development and training notebooks are located in the `dev/` directory:
- `knn_model.ipynb`: KNN model development and training
- `log_reg_model.ipynb`: Logistic Regression model development and training


## License

This project is for educational purposes.

## Author

Created by Subhstatix - Machine Learning Project
