# Spaceship Titanic Kaggle Competition

![Spaceship Titanic](https://live.staticflickr.com/2258/2502603301_57c6af2a9a_z.jpg)

This repository contains my submission for the **Spaceship Titanic** Kaggle competition. The goal of the competition is to predict which passengers were transported to an alternate dimension during the Titanic's fateful space voyage.

## Overview

### Key Scripts and Notebooks

- **`eda.ipynb`:** Conducts exploratory data analysis (EDA) to understand the dataset.
- **`logres.py`:** Trains and validates a logistic regression model using stratified K-fold cross-validation.
- **`submission.py`:** Generates the final `submission.csv` file for Kaggle.

### Results

- **Kaggle Public Leaderboard Score:** 0.79191

### How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spaceship-titanic.git
   cd spaceship-titanic
   ```

2. Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the data with stratified K-folds (if applicable):

   ```bash
   python src/create_folds.py
   ```

4. Train the model:

   ```bash
   python src/logres.py
   ```

5. Generate the submission:

   ```bash
   python src/submission.py
   ```

## License

This project is licensed under the MIT License.

---

### **Conclusion:**

This approach keeps the README concise and focused on what matters most—showing that you've completed the project, how to run it, and what results you achieved. It also saves you from the need to update the README frequently as the project evolves. This is often more than sufficient for a portfolio project where the primary audience is someone reviewing your work to assess your skills.
