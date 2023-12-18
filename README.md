# ML ZoomCamp Capstone Project 1

This is the repository for the ML ZoomCamp Capstone Project 1. In this project, I worked on a machine learning problem that aims to predict medical costs for individuals based on various factors such as age, BMI, smoking status, and region. The project is organized as follows:

## Project Overview

The goal of this project is to develop a machine learning model that can accurately predict medical costs for individuals based on various factors such as age, BMI, smoking status, and region. By accurately predicting medical costs, insurance companies can better estimate premiums and individuals can make more informed decisions about their healthcare.

## Dataset

The dataset used for this project is the Medical Cost Personal Datasets available on Kaggle. It contains information about individuals' medical costs, as well as their demographic and lifestyle attributes. The dataset is provided in a CSV format and can be found in the `data/` directory under notebook folder of this repository as `insurance.csv`.

## Project Structure

The project is organized as follows:

- `notebooks/`: Under data folder, consists the source data. This notebook folder contains the code for EDA and data preprocessing. This notebook contains the code for model selection, and performance evaluation as well.
- `src/`: This directory contains source code for utility functions and helper classes. Also contains the components and pipeline definition for the project.
- `templates/`: This directory contains the HTML templates for the web application.
- `app.py`: This file contains the code for the Flask web application.
- `test_app.py`: This file contains the tests for the Flask web application.
- `Dockerfile`: This file contains the instructions for building the Docker image.
- `requirements.txt`: This file contains the list of Python dependencies for the project.
- `README.md`: This file provides an overview of the project and its structure.

## Getting Started

To get started with the project, best way is to look at the notebooks in the `notebooks/` directory. These notebooks provide step-by-step instructions on data preprocessing, feature engineering, model selection, and performance evaluation.

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/capstone-project1.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Explore the Jupyter notebooks in the `notebooks/` directory to understand the project workflow.
4. Run the notebooks in the specified order to reproduce the results.

## Dependencies

The project requires the following dependencies to run:

- Python 3.8
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- ipykernel

Above are some of the dependencies. Please refer to the `requirements.txt` file for the complete list of dependencies.

## Installation and Environment Setup

To install the required dependencies and set up the environment, follow these steps:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/capstone-project1.git`
2. Change to the project directory: `cd MLCapstone1`
3. Create a virtual environment: `python -m venv env`
4. Activate the virtual environment:
   - For Windows: `.\env\Scripts\activate`
   - For macOS/Linux: `source env/bin/activate`
5. Install the required dependencies: `pip install -r requirements.txt`

## Deployment

## Containerization

Refer to the `Dockerfile` for instructions on how to build the Docker image.

### Google Cloud Deployment

To deploy the service to Google Cloud, follow these steps:

1. Sign up for a Google Cloud account.
2. Set up a project and enable the necessary APIs (e.g., Cloud Run, Artifact Registry and others).
3. Deploy the Docker image to Google Cloud Run.
4. Access the service using the provided URL.

For detailed instructions and code examples, please refer to the blog post [here](https://lesliemwubbel.com/setting-up-a-flask-app-and-deploying-it-via-google-cloud/).

## Testing

To test the deployed service, follow these steps:

1. Already a test_app.py file is available in the repository. Run the test_app.py file to test the flask app.
2. Verify the response and check for any errors or issues.
3. Optionally, go to the assets folder and see screenshots of the test results.

## Model Training and Evaluation

The model training and evaluation process is documented in the Jupyter notebooks in the `notebooks/` directory. The notebooks provide step-by-step instructions on data preprocessing, feature engineering, model selection, hyperparameter tuning, and performance evaluation.

## Results

The trained model with hyperparameter tuning achieved an accuracy of 89% on the test set. The model's performance was evaluated using metrics such as mean absolute error (MAE), mean squared error (MSE), and R-squared.

## Conclusion

In this project, we successfully developed a machine learning model that can accurately predict medical costs for individuals. The model can be used by insurance companies to estimate premiums and individuals to make informed decisions about their healthcare.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or suggestions regarding the project, feel free to reach out to me head over to my [GitHub profile](https://github.com/sitamgithub-MSIT)

Happy coding!
