# Craigslist Post Category Prediction

## Important Links Related to Craigslist Post Category Prediction

Craigslist Post Category Prediction App: [https://ai-based-post-categorization-prasetyo-widyantoro.streamlit.app/](https://ai-based-post-categorization-prasetyo-widyantoro.streamlit.app/)

Project Documentation: [https://prasetyowidyantoro.github.io/Documentation-for-Automated-Ad-Category-Prediction/](https://prasetyowidyantoro.github.io/Documentation-for-Automated-Ad-Category-Prediction/)


==============================

Craigslist Post Category Prediction aims to predict the category of each post published on the Craigslist classified ads platform. 


Project Organization
------------

    ├── config
    │   ├── config.yaml                 <- Configuration file that defines various paths and settings for the data project.
    ├── README.md                       <- The top-level README for developers using this project.
    ├── data
    │   ├── raw                         <- The original, immutable data dump.
    │   ├── processed                   <- Intermediate data that has been transformed.
    │   └── final                       <- The final, canonical data sets for modeling.
    ├── docker
    │   ├── api                         <- Dockerfile and requirements for the API service.
    │   │   ├── dockerfile              <- Docker file used for building the API service container.
    │   │   └── requirement.txt         <- Requirements or libraries used to run the API service.
    │   ├── streamlit                   <- Dockerfile and requirements for the Streamlit UI.
    │   │   ├── dockerfile              <- Docker file used for building the Streamlit UI container.
    │   │   └── requirement.txt         <- Requirements or libraries used to run the Streamlit UI.
    ├── docs 
    │   ├── docs             
    │   │   ├── images                  <- Images or screenshots resulting from data processing.
    │   │   ├── about.md                <- Information about the creator of the project.
    │   │   └── index.md                <- Documentation of the Craigslist Machine Learning project.
    │   ├── mkdocs.yaml                 <- Mkdocs configuration file.
    ├── model                           <- Trained and serialized models, model predictions, or model summaries.
    ├── notebook experiment             <- Directory for storing experimental notebooks.
    ├── src
    │   ├── api.py                       <- API service script.
    │   ├── function_1_data_pipeline.py  <- Data pipeline function.
    │   ├── function_2_data_processing.py<- Data processing function.
    │   ├── function_3_modeling.py       <- Modeling function.
    │   ├── streamlit-deeplearning.py    <- Streamlit UI script.
    │   └── util.py                      <- Utility script.
    ├── training_log                     <- Folder for storing training logs.
    ├── .dockerignore                    <- Docker ignore file.
    ├── .gitattributes                   <- Git attributes file.
    ├── .gitignore                       <- Git ignore file.
    ├── 1. Data Preparation.ipynb        <- Jupyter notebook for data preparation.
    ├── 2. EDA.ipynb                     <- Jupyter notebook for exploratory data analysis.
    ├── 3. Data Preprocessing.ipynb      <- Jupyter notebook for data preprocessing.
    ├── 4. Training Data TFIDF.ipynb     <- Jupyter notebook for training data using TFIDF.
    ├── docker-compose.yaml              <- Docker Compose configuration file.
    └── requirements.txt                 <- The requirements file for reproducing the analysis environment.
                                           Generated with `pip freeze > requirements.txt`.

--------

## Initial Configuration

1. Create a folder to clone the files available on this GitHub page.
2. Clone this GitHub repository.
3. Set up a virtual environment.
4. Activate the created virtual environment.
5. Install the requirements by running `pip install -r requirements.txt`.
6. The files are configured; proceed with the following steps as desired.

## Setting Up "Craigslist Post Category Prediction" with Docker

To set up the "Craigslist Post Category Prediction" using Docker, follow these steps:

1. Ensure that you have cloned the repository for this program.
2. Install a virtual environment.
3. Activate the created virtual environment.
4. Install the required packages listed in `requirements.txt` by executing:
   ```
   pip install -r requirements.txt
   ```
5. Make sure Docker is activated and that you are logged in.
6. Run the following script in the terminal with the activated virtual environment:
   ```
   docker compose up -d --build
   ```
7. The "Craigslist Post Category Prediction" Machine Learning Service is now ready for use.
8. Access the API documentation at [http://localhost:8080/docs](http://localhost:8080/docs) and the UI Front End (Streamlit) at [http://localhost:8501](http://localhost:8501).

## End-to-End Machine Learning Process

The process will encompass data preparation, Exploratory Data Analysis (EDA), Data Preprocessing, Data Modeling, and culminate in a Machine Learning Service using APIs. The entire system will be integrated into a Streamlit application with Docker for containerization, and the deployment will be facilitated through the Streamlit Share server.

### Data Preparation

**Data Preparation Architecture Diagram**
![1_multiclass_prep](docs/docs/images/1_multiclass_prep.png)

The data preparation process begins with reading raw data, followed by the definition of columns to be processed. This includes data validation, specifically checking data types, identifying duplicate data, and splitting the data into training, validation, and test sets. Finally, the processed data is saved before proceeding to the data preprocessing stage.

#### Dataset Definition

The data used in this analysis includes information about various services, community activities, and properties advertised on the Craigslist e-commerce platform. The dataset contains information from sixteen different cities. Here is a summary of the dataset:

Data Source: Craigslist

The dataset consists of 20,217 rows with four main columns:

1. **city**: This column includes the name of the city where the advertisement is posted.
2. **category**: This column contains the category or classification of the advertisement, reflecting the type of service, item, or property offered.
3. **section**: This column indicates the section of the Craigslist platform where the advertisement is posted, such as services, community, housing, or for-sale.
4. **heading**: This column contains the title or a brief description of the advertisement, providing an overview of the services, items, or properties offered.

Subsequently, we can perform further analysis related to this information to understand trends and patterns on the Craigslist platform.

### Exploratory Data Analysis (EDA)

**Exploratory Data Analysis (EDA) Architecture Diagram**
![2_multiclass_EDA](docs/docs/images/2_multiclass_EDA.png)

Next is the Exploratory Data Analysis (EDA) phase. In this stage, exploration of the data is conducted to gain a deeper understanding. The process includes several steps. Firstly, Basic Text Exploration is performed, such as finding the "heading" with the longest and shortest names, examining products with titles containing specific characteristics, searching for rows containing punctuation or special characters, and finding rows with multiple spaces in the title column. Secondly, Data Proportion analysis is carried out by checking the proportion of data in terms of both the count and percentage of target labels. Thirdly, Text Representation involves representing words in text data. Lastly, a WordCloud is generated to visualize word frequency.

### Data Preprocessing

**Data Preprocessing Architecture**
![3_multiclass_preproc](docs/docs/images/3_multiclass_preproc.png)

Moving on to the Data Preprocessing stage, the initial step after reading the data prepared in the Data Preparation phase is the execution of post-EDA (Mandatory from EDA) actions. This includes operations such as removing special characters, converting text to lowercase, removing stopwords, and joining the words with a single space. Following this, the data is transformed using TFIDF to assign weights to unique and important words in a document, differentiating one document from another. Additionally, label encoding is applied to the target data with the following mapping: activities -> 0, appliances -> 1, artists -> 2, automotive -> 3, cell-phones -> 4, childcare -> 5, general -> 6, household-services -> 7, housing -> 8, photography -> 9, real-estate -> 10, shared -> 11, temporary -> 12, therapeutic -> 13, video-games -> 14, wanted-housing -> 15. After completing these processes, the data is saved.

### Data Modeling

**Data Modeling Architecture**
![4_multiclass_modeling](docs/docs/images/4_multiclass_modeling.png)

Once the data from the preprocessing stage is saved, it is loaded into the data modeling phase. Subsequently, the data undergoes the modeling process with experiments using various models such as extratrees, xgboost, gradient boosting, SVC, logistic regression, lightgbm, and deep learning. However, the best results, with the optimal evaluation metric values, are obtained using the deep learning model, achieving a Test recall of 0.86.

## **Neural Network Architecture:**
The neural network is named `AdvancedNN`, designed for a classification problem. It consists of an input layer, a hidden layer with 256 neurons using the ReLU activation function, a dropout layer for regularization, and an output layer. The model's architecture is crucial in capturing complex patterns in the input data.

**Model Initialization and Optimization:**
The model is initialized with the specified architecture. The chosen loss function is CrossEntropyLoss, suitable for multi-class classification tasks. The Adam optimizer is used to update the model parameters, and a learning rate of 0.001 is set. Additionally, a learning rate scheduler (ReduceLROnPlateau) is employed to adjust the learning rate dynamically during training based on the model's performance on the validation set.

**Training Process:**
The training process involves iterating through the dataset for a specified number of epochs (50 in this case). During each epoch, the model is trained on the training dataset using backpropagation to minimize the defined loss. The Adam optimizer updates the model's parameters. The training loop also includes a validation step to assess the model's performance on a separate validation dataset.

**Early Stopping:**
To prevent overfitting, an early stopping mechanism is implemented. The validation loss is monitored during training, and if no improvement is observed for a predefined number of epochs (5 in this case), the training is halted early. The best-performing model is saved, ensuring that the model with the lowest validation loss is retained.

**Theoretical Explanation:**
- **Neural Network Architecture:** The neural network architecture follows a standard feedforward design. The ReLU activation function introduces non-linearity to capture complex relationships within the data. The dropout layer helps prevent overfitting by randomly setting a fraction of input units to zero during training.
  
- **Loss Function and Optimization:** CrossEntropyLoss is commonly used for multi-class classification as it measures the dissimilarity between predicted and actual class distributions. The Adam optimizer efficiently adjusts the model's weights to minimize this loss.

- **Training Loop:** The model is trained iteratively on batches of data. The learning rate scheduler adjusts the learning rate based on the model's performance on the validation set, allowing for better convergence.

- **Early Stopping:** Early stopping is a regularization technique that halts training when the model's performance on the validation set ceases to improve. This prevents the model from becoming overly specialized to the training data and enhances generalization to unseen data.

In summary, the provided code implements a robust training framework for an ANN, incorporating architectural choices, optimization strategies, and regularization techniques to achieve accurate and generalizable results.

**Classification Report**

![Modeling_3_Classification_Report](docs/docs/images/Modeling_3_Classification_Report.png)

## Machine Learning Services Architecture

![5_multiclass_mlservice](docs/docs/images/5_multiclass_mlservice.png)

- The machine learning service process begins with user input, where the provided data is converted into a Pandas dataframe. The input data undergoes a data defense process to ensure compatibility with the pre-trained model. Subsequently, special characters are removed, and the text is converted to lowercase. Stopwords are then removed, and the remaining words are joined with a single space. The cleansed data undergoes transformation using TFIDF, and finally, predictions are made using the pre-trained model saved during the data modeling phase.

- During the prediction phase, users input data through the Streamlit web interface, serving as the front end of the application. Upon the user's click on "Predict," the data is sent as a request to the back end. The back end processes the request and responds with the prediction results to the user.

- This architecture ensures a seamless interaction between the user interface, data preprocessing, and model prediction, providing an end-to-end machine learning service for efficient and user-friendly predictions. The API framework employed in the Machine Learning Service process is FASTAPI, and the front end utilizes Streamlit.

- Following the various processes and saving the model, the next step involves deployment using API and Streamlit. The image below provides an example of accessing ML Services through FASTAPI Swagger UI.

Here is an example of input data used to access the API:

**Prediction using API with FASTAPI**

![API_1_FastApi](docs/docs/images/API_1_FastApi.png)

- This demonstrates the seamless interaction between the user interface, data preprocessing, model prediction, and the API, providing a user-friendly and accessible method for making predictions. To enhance user-friendliness and provide a more powerful interface, users are presented with a simple application built using Streamlit services. Here's an example of how it can be utilized.

**Prediction using provided form**

![ML_Service_1_Streamlit](docs/docs/images/ML_Service_1_Streamlit.png)

**Prediction using JSON file**

![ML_Service_2_Streamlit](docs/docs/images/ML_Service_2_Streamlit.png)

## Conclusion

After thorough experimentation, the deep learning neural network emerged as the top-performing model, achieving an impressive recall of 0.86. Notably, the satisfactory results were attained without the need for data augmentation or balancing.

The Craigslist Post Category Prediction project embodies a comprehensive end-to-end machine learning solution, encompassing vital stages such as data preparation, exploratory data analysis, data preprocessing, data modeling, and machine learning services. The implementation strategically employs a deep learning neural network to predict and categorize posts on the Craigslist platform, placing a strong emphasis on achieving precision in categorization.

The selected model is fine-tuned to optimize for high recall, ensuring accurate detection of products genuinely belonging to specific categories. The neural network architecture, loss function, optimization strategy, training loop, and early stopping mechanism are meticulously crafted to yield accurate and generalizable results.

The integration of machine learning services through API and Streamlit provides users with a seamless and user-friendly interface for inputting data and receiving predictions effortlessly. Furthermore, the Dockerization of services using Docker Compose facilitates straightforward deployment and scalability.

In essence, the Craigslist Post Category Prediction project serves as a testament to the transformative power of machine learning in enhancing user experience. It establishes an efficient and intuitive system for organizing and categorizing classified ads on the Craigslist platform.

## Further Research

1. **Exploration with Pre-trained Models:**
   Conducting further exploration using advanced pre-trained models such as BERT, GPT, and others.
   
3. **Evaluation and Application of Data Balancing Techniques:**
   Assessing and implementing data balancing techniques like oversampling or undersampling to address class imbalances.

5. **Collaboration with Domain Experts:**
   Involving collaboration with domain experts and stakeholders to ensure the model adds value in relevant contexts.

These proposed steps aim to enhance the model's performance, explore advanced methodologies, and ensure applicability in real-world scenarios through collaborative efforts.

## References

- [Multi-Class Classification in Python Example](https://www.projectpro.io/article/multi-class-classification-python-example/547?source=post_page-----81975d03e4a3--------------------------------)
- [Comprehensive Guide to MultiClass Classification with Scikit-Learn](https://towardsdatascience.com/comprehensive-guide-to-multiclass-classification-with-sklearn-127cc500f362?source=post_page-----81975d03e4a3--------------------------------)
- [Binary MultiClass Classification using Scikit-Learn](https://www.kaggle.com/code/satishgunjal/binary-multiclass-classification-using-sklearn?source=post_page-----81975d03e4a3--------------------------------#Train-and-Evaluate-a-Binary-Classification-Model)
- [Understanding Multi-Class Classification](https://builtin.com/machine-learning/multiclass-classification)
- Iskandar Zulkarnain Maulana Putra, T., Farhan Bukhori, A., Ilmu Pengetahuan Alam, dan, & Gadjah Mada, U. (2022). [Model Klasifikasi Berbasis Multiclass Classification dengan Kombinasi Indobert Embedding dan Long Short-Term Memory untuk Tweet Berbahasa Indonesia](https://doi.org/10.35912/jisted.v1i1.1509) (Classification Model Based on Multiclass Classification with a Combination of Indobert Embedding and Long Short-Term Memory for Indonesian-language Tweets). *Jurnal Ilmu Siber Dan Teknologi Digital (JISTED)*, 1(1), 1–28.
- Nugroho, W. H., Handoyo, S., Akri, Y. J., & Sulistyono, A. D. (2022). [Building Multiclass Classification Model of Logistic Regression and Decision Tree Using the Chi-Square Test for Variable Selection Method](https://doi.org/10.55463/issn.1674-2974.49.4.17). *Journal of Hunan University Natural Sciences*, 49(4), 172–181.
- Rabbimov, I. M., & Kobilov, S. S. (2020). [Multi-Class Text Classification of Uzbek News Articles using Machine Learning](https://doi.org/10.1088/1742-6596/1546/1/012097). *Journal of Physics: Conference Series*, 1546(1).