# 📄✨ Resume Screening AI-based System ✨📄

Welcome to the **Resume Screening AI-based System**! This project is a Flask web application designed to automate the process of resume screening. It leverages machine learning models to categorize resumes and recommend suitable job roles based on the content of the resumes. Additionally, it extracts key information such as contact details, skills, and education.

## 🚀 Features

- 📂 Upload and parse resumes in PDF or TXT format.
- 🔍 Clean and preprocess resume text.
- 🏷️ Categorize resumes into predefined categories.
- 💼 Recommend job roles based on resume content.
- 📞 Extract contact numbers and email addresses.
- 🛠️ Extract skills and educational qualifications.
- 🖥️ Simple and intuitive web interface.

## 🛠️ Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/resume-screening-ai.git
    cd resume-screening-ai
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the pre-trained models** and place them in the project directory:
    - `rf_classifier_categorization.pkl`
    - `tfidf_vectorizer_categorization.pkl`
    - `rf_classifier_job_recommendation.pkl`
    - `tfidf_vectorizer_job_recommendation.pkl`

## 📦 Usage

1. **Run the Flask application**:
    ```sh
    python app.py
    ```

2. **Open your browser** and navigate to:
    ```
    http://127.0.0.1:5000
    ```

3. **Upload a resume** in PDF or TXT format.

4. **Get the results**:
    - Predicted category 🏷️
    - Recommended job role 💼
    - Extracted contact number 📞
    - Extracted email address 📧
    - Extracted skills 🛠️
    - Extracted education 🎓



## 🤖 How It Works

- **PDF/Text Parsing**: The application uses `PyPDF2` to extract text from PDF files.
- **Text Cleaning**: The extracted text is cleaned using regular expressions to remove unwanted characters and symbols.
- **Categorization**: A pre-trained Random Forest model categorizes the resume into a specific category.
- **Job Recommendation**: Another pre-trained Random Forest model recommends suitable job roles.
- **Information Extraction**: Regular expressions are used to extract contact numbers, email addresses, skills, and education.

## 👨‍💻 Contributing

Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



Happy Screening! 🎉
