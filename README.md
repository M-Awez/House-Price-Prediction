# 🏠 House Price Prediction using Multiple Linear Regression

A Machine Learning web application that predicts the **estimated price of a house** based on different property-related features using **Multiple Linear Regression**.

The project is implemented using **Python, Pandas, NumPy, Scikit-Learn, Flask, HTML/CSS**, and is deployed online using **Render**.

---

## 📌 Project Overview

House prices depend on multiple factors such as the number of bedrooms, bathrooms, living area, lot size, location, house condition, construction year, and other property characteristics.

This project uses **Multiple Linear Regression (MLR)** to learn the relationship between these independent variables and the target house price.

The trained machine learning model is integrated into a **Flask web application**, where users can enter property details through a web form and receive an estimated house price.

### 🔄 Overall Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Multiple Linear Regression
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model as model.pkl
   ↓
Flask Web Application
   ↓
User Enters Property Details
   ↓
Prediction
   ↓
Estimated House Price
```

---

# 🎯 Objectives

The main objectives of this project are:

* To understand and implement Multiple Linear Regression.
* To preprocess real-world housing data.
* To extract useful features from the date column.
* To convert categorical variables into numerical representations.
* To divide the dataset into training and testing data.
* To train a Linear Regression model.
* To evaluate the model using accuracy/R² score and loss.
* To save the trained model using Pickle.
* To integrate the trained model with Flask.
* To create a web interface for house price prediction.
* To deploy the application using Render.

---

# ✨ Features

* 🏠 House price prediction
* 🤖 Machine Learning based prediction
* 📊 Multiple Linear Regression
* 📅 Date feature extraction
* 🌎 Country encoding
* 🏙️ City encoding
* 📈 Training and testing evaluation
* 💾 Model serialization using Pickle
* 🌐 Flask web application
* 🎨 Responsive web interface
* 🚀 Render deployment
* ⚡ Real-time prediction through a web form

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Machine Learning

* Scikit-Learn
* Multiple Linear Regression

## Data Processing

* Pandas
* NumPy

## Data Visualization

* Matplotlib

## Web Framework

* Flask

## Model Serialization

* Pickle

## Frontend

* HTML
* CSS

## Deployment

* Render

---

# 📊 Dataset

The project uses a housing dataset containing **more than 4,500 records** and multiple columns describing different properties.

The dataset contains property-related information such as:

* Date
* Bedrooms
* Bathrooms
* Living Area
* Lot Area
* Floors
* Waterfront
* View
* Condition
* Above Ground Area
* Basement Area
* Year Built
* Year Renovated
* City
* Country
* House Price

The exact columns used by the model depend on the dataset supplied to the project.

---

# 🧹 Data Preprocessing

Several preprocessing operations are performed before training the machine learning model.

## 1. Reading the Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv('data.csv')
```

---

## 2. Date Conversion

The `date` column is converted into a Pandas datetime object:

```python
df['date'] = pd.to_datetime(
    df['date'],
    dayfirst=False,
    errors='coerce'
)
```

This allows individual date components to be extracted.

---

## 3. Feature Engineering

The date is divided into three separate features:

```python
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
```

Therefore, the original date is transformed into:

* `day`
* `month`
* `year`

The original `date` column is then removed:

```python
df = df.drop(['date'], axis=1)
```

---

# 🌎 Country Encoding

The country column is converted from categorical data into numerical data.

The current implementation maps:

```python
df['country'] = df['country'].map({
    'USA': 1
})
```

Therefore:

```text
USA → 1
```

This allows the categorical value to be processed by the regression model.

---

# 🏙️ City Encoding

The unique cities are extracted:

```python
l = df['city'].unique()
```

A numerical mapping is then created for each city:

```python
d = {}

for i in range(len(l)):
    d[l[i]] = i
```

The city column is converted using this mapping:

```python
df['city'] = df['city'].map(d)
```

This converts the city names into numerical values that can be used by the machine learning model.

---

# 🎯 Feature and Target Selection

The dataset is separated into independent variables and the target variable.

```python
self.X = df.iloc[:, 1:]
self.y = df.iloc[:, 0:1]
```

### Independent Variables

The independent variables represent the property characteristics used to predict the house price.

### Target Variable

The first column of the processed dataset is used as the target variable.

```text
X → Input Features
y → House Price
```

---

# 📚 Train-Test Split

The dataset is divided into training and testing data using Scikit-Learn:

```python
X_train, X_test, y_train, y_test = train_test_split(
    self.X,
    self.y,
    train_size=0.8,
    random_state=42
)
```

The split uses:

* **80% → Training data**
* **20% → Testing data**
* `random_state = 42`

The training dataset is used to teach the model, while the testing dataset is used to evaluate its performance on unseen data.

---

# 🧠 Machine Learning Model

The project uses:

## Multiple Linear Regression

Multiple Linear Regression predicts a dependent variable based on multiple independent variables.

The general equation is:

```text
y = b₀ + b₁x₁ + b₂x₂ + b₃x₃ + ... + bₙxₙ
```

Where:

* `y` = predicted house price
* `b₀` = intercept
* `b₁ ... bₙ` = model coefficients
* `x₁ ... xₙ` = input features

The model is created using:

```python
self.reg = LinearRegression()
```

and trained using:

```python
self.reg.fit(a, b)
```

---

# 🏋️ Model Training

The model is trained using the training dataset:

```python
obj.train(X_train, y_train)
```

Internally, this executes:

```python
self.reg.fit(a, b)
```

After training, the model can generate predictions for new data.

---

# 🔮 Prediction

Predictions are generated using:

```python
self.reg.predict(a)
```

The project generates predictions for both the training and testing datasets.

### Training Prediction

```python
train_prds = obj.predictions(X_train)
```

### Testing Prediction

```python
test_prds = obj.predictions(X_test)
```

---

# 📈 Model Evaluation

The project evaluates the model using a custom `accuracy_loss()` function.

The implementation calculates:

### R² Score

The calculation follows:

```text
R² = 1 - (Sum of Squared Errors / Total Sum of Squares)
```

In the code:

```python
acc = 1 - (n / d)
```

The result is printed as a percentage:

```python
print(f"Accuracy: {acc * 100}")
```

---

# 📉 Loss Calculation

The project also calculates the Root Mean Squared Error (RMSE):

```python
loss = (n / len(y_train)) ** (1 / 2)
```

The loss is printed using:

```python
print(f"loss: {loss}")
```

A lower RMSE indicates that the predicted values are, on average, closer to the actual values.

---

# 💾 Saving the Trained Model

After training, the Linear Regression model is saved using Python's Pickle library.

```python
with open("model.pkl", "wb") as f:
    pickle.dump(obj.reg, f)
```

This creates:

```text
model.pkl
```

The saved model can then be loaded by the Flask application without retraining every time the server starts.

---

# 🌐 Flask Web Application

The trained model is integrated into a Flask application.

The model is loaded when the Flask application starts:

```python
with open('model.pkl', 'rb') as t:
    m = pickle.load(t)
```

The Flask application is created using:

```python
app = Flask(__name__)
```

---

# 🛣️ Flask Routes

## Home Route

```python
@app.route('/')
def index():
    return render_template('index.html')
```

The `/` route displays the prediction webpage.

---

## Prediction Route

```python
@app.route('/predict', methods=["POST", "GET"])
def predict():
```

The `/predict` route receives the values submitted through the HTML form.

The form values are collected using:

```python
l = list(request.form.values())
```

The numerical values are converted into floating-point numbers:

```python
a = [float(i) for i in l[1:]]
```

---

# 📅 Processing the User's Date

The date entered by the user is converted using:

```python
date = pd.to_datetime(
    l[0],
    dayfirst=False,
    errors='coerce'
)
```

The application extracts:

```python
date.day
date.month
date.year
```

These values are appended to the prediction input:

```python
a.append(date.day)
a.append(date.month)
a.append(date.year)
```

The resulting feature array is passed to the trained model.

---

# 🔮 Generating the Final Prediction

The Flask application sends the input data to the saved Linear Regression model:

```python
s = m.predict([a])
```

The prediction is then sent back to the HTML page:

```python
return render_template(
    'index.html',
    result=float(s[0])
)
```

The estimated house price can then be displayed to the user.

---

# 🎨 User Interface

The frontend is implemented using HTML and CSS.

The application provides input fields for the property information required by the machine learning model.

The interface includes:

* Property date
* Number of bedrooms
* Number of bathrooms
* Living area
* Lot area
* Floors
* Waterfront
* View rating
* House condition
* Above-ground area
* Basement area
* Year built
* Year renovated
* City
* Country

The UI uses a modern dark **glassmorphism design** with blue and purple gradients.

---

# 📁 Project Structure

A typical project structure is:

```text
House-Price-Prediction/
│
├── app.py
├── model.py
├── data.csv
├── model.pkl
├── requirements.txt
├── templates/
│   └── index.html
│
└── README.md
```

### File Description

| File               | Description                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `model.py`         | Loads the dataset, preprocesses the data, trains the Linear Regression model and saves it |
| `app.py`           | Flask backend for serving the web application and generating predictions                  |
| `data.csv`         | Housing dataset                                                                           |
| `model.pkl`        | Serialized trained Linear Regression model                                                |
| `index.html`       | Frontend prediction interface                                                             |
| `requirements.txt` | Python dependencies                                                                       |
| `README.md`        | Project documentation                                                                     |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
```

Move into the project directory:

```bash
cd house-price-prediction
```

---

# 🐍 Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
numpy
pandas
matplotlib
scikit-learn
Flask
gunicorn
```

---

# ▶️ Run the Machine Learning Model

Before running the Flask application, train the model:

```bash
python model.py
```

After successful training, the following file will be generated:

```text
model.pkl
```

---

# ▶️ Run the Flask Application

Start the Flask application:

```bash
python app.py
```

The application will run locally.

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

---

# 🔄 Prediction Process

The complete prediction process is:

```text
User Opens Website
        ↓
Enters Property Details
        ↓
Clicks "Predict House Price"
        ↓
Flask Receives Form Data
        ↓
Date is Converted
        ↓
Day / Month / Year Extracted
        ↓
Input Features Prepared
        ↓
Saved Linear Regression Model Loaded
        ↓
Model Generates Prediction
        ↓
Estimated House Price Displayed
```

---

# 🚀 Deployment

The application has been deployed using **Render**.

For deployment, the project requires a production WSGI server such as Gunicorn.

A typical Render start command is:

```bash
gunicorn app:app
```

The deployed application can then be accessed through the Render-provided URL.

---

# 📋 Requirements

A `requirements.txt` file can contain the packages required by the application.

Example:

```text
Flask
gunicorn
numpy
pandas
scikit-learn
matplotlib
```

It is recommended to use compatible package versions when deploying the saved `model.pkl` file.

---

# ⚠️ Important Notes

The model and Flask application must use the **same feature order**.

The preprocessing performed during model training should correspond exactly to the preprocessing performed when receiving prediction inputs.

In particular:

* Date must be converted into day, month, and year.
* Country must use the same numerical mapping.
* City must use the same numerical mapping.
* Input features must be provided in the same order used during training.

This is important because a machine learning model expects the input features in the same structure that it learned during training.

---

# 🔐 Model Serialization

The project uses Pickle to serialize the trained model:

```python
pickle.dump(obj.reg, f)
```

The Flask application then loads the model:

```python
pickle.load(t)
```

This allows the application to make predictions without retraining the model every time.

---

# 📊 Model Evaluation

The project evaluates both:

### Training Dataset

```text
X_train → Linear Regression → Training Predictions
```

### Testing Dataset

```text
X_test → Linear Regression → Testing Predictions
```

For each dataset, the project calculates:

* R²-based accuracy percentage
* RMSE loss

This provides an indication of how well the regression model fits the data and performs on unseen records.

---

# 🔮 Future Enhancements

Possible improvements for future versions include:

* Add feature scaling where appropriate.
* Use a more robust categorical encoding method for cities.
* Store the city encoding mapping for use during prediction.
* Add validation for user input.
* Display additional model evaluation metrics.
* Add graphs showing actual vs predicted prices.
* Compare Multiple Linear Regression with other regression algorithms.
* Add Random Forest Regression.
* Add Gradient Boosting Regression.
* Add XGBoost Regression.
* Add a prediction history feature.
* Improve error handling.
* Add a database for storing predictions.
* Add user authentication.
* Improve the deployment configuration.
* Add model retraining functionality.
* Add API endpoints for external applications.

---

# 🎓 Learning Outcomes

Through this project, the following concepts were practiced:

* Python programming
* Object-oriented programming
* Pandas data preprocessing
* NumPy operations
* Date-time feature extraction
* Categorical data encoding
* Train-test splitting
* Multiple Linear Regression
* Model training
* Model prediction
* R² score calculation
* RMSE calculation
* Pickle model serialization
* Flask application development
* HTML/CSS frontend development
* Machine Learning model integration
* Web application deployment
* Render deployment

---

# 👨‍💻 Author

**Your Name**

Computer Science Student / Machine Learning Enthusiast

### Technologies & Interests

* Python
* Machine Learning
* Data Science
* Flask
* Web Development
* Artificial Intelligence

---

# ⭐ Project Highlights

```text
🏠 House Price Prediction
🤖 Multiple Linear Regression
📊 4,500+ Dataset Records
🐍 Python
🌐 Flask
📈 Scikit-Learn
💾 Pickle
🎨 HTML + CSS
🚀 Render Deployment
```

---

# 📄 License

This project is created for **educational and academic purposes**.

You are free to study, modify, and improve the project for learning purposes.
