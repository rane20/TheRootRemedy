# The Root Remedy Collaboration

## 📖 Purpose  

The **The Root Remedy** team aims to improve user experience by enhancing the chatbot's ability to understand user queries. By integrating sentiment and intent analysis, we sought to provide personalized, empathetic, and contextually relevant responses.  

This project focused on:  
- Building a domain-specific dataset for gut health, wellness, skin health, and beauty.
- Perfoming sentiment analysis
- Implementing a BERT-based model to predict **sentiment** and **intent**.  
- Integrating the model with the Root Remedy chatbot to improve its responsiveness.  

---

## 📚 Table of Contents  

1. [Purpose](#-purpose)  
2. [Project Overview, Objectives, and Goals](#-project-overview-objectives-and-goals)  
3. [Methodology](#-methodology)  
4. [Results and Key Findings](#-results-and-key-findings)  
5. [Visualizations](#-visualizations)  
6. [Potential Next Steps](#-potential-next-steps)  
7. [Installation](#-installation)  
8. [Usage](#-usage)  
9. [Contributing](#-contributing)  
10. [License](#-license)  
11. [Credits and Acknowledgments](#-credits-and-acknowledgments)  

---

## 🔨 Project Overview, Objectives, and Goals  

### Objectives:  
1. Develop a robust dataset tailored to user queries about gut health, wellness, and skin health.  
2. Train a BERT-based model for dual tasks: sentiment analysis and intent detection.  
3. Improve chatbot functionality with these advanced AI capabilities.  

### Problem Addressed:  
Existing chatbots often lack domain-specific understanding and struggle with emotional nuance. This project addresses these gaps by enabling the chatbot to interpret both the sentiment and intent behind user queries.  

---

## 📊 Methodology  

### Data Collection and Preprocessing:  
- **Keyword Research**: Identified high-value terms in gut health, wellness, skin health, beauty, and general inquiries.  
- **Data Generation**: Created diverse user queries and annotated them with sentiment and intent labels.  
- **Preprocessing**: Tokenized sentences, removed noise, and converted labels into numerical formats.  

### Model Implementation:  
- Used Hugging Face’s **transformers** library to fine-tune a pre-trained BERT model.  
- Tasks:
  - **Dataset Creation**: Creating dataset from scratch by utilizing a random sentence generator
  - **Sentiment Analysis**: Classified queries as positive or negative 
  - **Intent Detection**: Predicted user intents, such as gut health inquiries, skin health inquiries, wellness inquiries, gut supplement inquiries, and general inquiries. 

### Tools and Libraries:  
- **Python**: A versatile programming language used for data manipulation, model development, and orchestration of the machine learning pipeline.  

#### **Data Processing**  
- **Pandas**: Used for loading, cleaning, and preprocessing datasets. Provides easy-to-use data structures like DataFrames for organizing data.  
- **NumPy**: Used for efficient numerical computations, such as handling arrays and performing matrix operations (indirectly supports model training).  

#### **Machine Learning and NLP**  
- **AWS SageMaker**: A cloud-based machine learning service used for training and deploying machine learning models at scale.  
- **AWS Comprehend**: A natural language processing (NLP) service that provides pre-trained models for sentiment analysis, entity recognition, and intent detection, used to benchmark and annotate data.  
- **Hugging Face Transformers**:  
  - **BERT Model**: A transformer-based model used for feature extraction, fine-tuning, and building advanced NLP pipelines for sentiment and intent classification.  
  - **BertTokenizer**: Tokenizes text into subword tokens for BERT compatibility.  
  - **Trainer and TrainingArguments**: Simplifies fine-tuning transformer models by managing training loops, hyperparameter tuning, and evaluation.  

#### **Deep Learning Frameworks**  
- **PyTorch**: A deep learning library used for defining custom datasets, building models, and performing tensor operations.  
  - **torch.utils.data.Dataset**: Used to create a custom dataset class for PyTorch training.  
  - **torch.nn**: Provides modules for building neural networks, including fully connected layers and activation functions.  

#### **Model Training and Evaluation**  
- **scikit-learn**:  
  - **LabelEncoder**: Converts categorical labels into numerical form for machine learning models.  
  - **train_test_split**: Splits the dataset into training and testing sets to evaluate model performance.  

---

## 📈 Results and Key Findings  

### Dataset Results:
-**Samples**: 501 sentences
-**Sentiment**: Positive(260), Negative (268)
-**Intents**: Gut Health Inquiry(198), Gut Supplement Inquiry(10), Beauty Inquiry(78), Skin Health Inquiry(80), Wellness Inquiry(48), General Inquiry(86)

### Key Insights:  
- Sentiment predictions significantly improved chatbot empathy and response accuracy.  
- Intent detection allowed for more relevant recommendations and streamlined user interactions.
- The more data you process through the BERT model, the slower the model processes.
- Strive to have even examples for the dataset so that the model improves perfomance.

---

## 📊 Visualizations   
- Training/validation accuracy
![image](https://github.com/user-attachments/assets/fa019795-9198-4e9d-a5bc-5fc231d16d34)

- Example predictions with sentiment and intent classifications  
![image (1)](https://github.com/user-attachments/assets/076668f7-bd53-4624-a9c3-96c33712f93b)

---

## 🚀 Potential Next Steps  

- Improve on the dataset and create more samples.  
- Explore other transformer models for further performance gains.  

---

## 💻 Installation  

### Prerequisites:  
- Python 3.8 or higher  
- Required libraries: Hugging Face Transformers, Pandas, Pytorch

### Installation Steps:  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/rane20/TheRootRemedy.git  
  
---


## 🤝 Contributing  

We welcome contributions to this project! To contribute:  
1. Fork the repository.  
2. Create a new branch for your feature or bugfix.  
3. Submit a pull request with a clear description of changes.  

---

## 📜 License  

This project is licensed under the MIT License. See the LICENSE file for more details.  

---

## 🌟 Credits and Acknowledgments  

This project was a collaboration between **The Root Remedy** and the **Break Through Tech AI program** at UCLA. 
Special thanks to The Root Remedy team, specifically Marcella Graham(CEO), for their invaluable guidance and support. 
Special thanks to Swagath B for being our TA throughout the project and for providing their guidance and support.
Special thanks to Break Through Tech AI Program for providing the opportunities and experiences related to data science, machine learning, and AI. 
Additional acknowledgments go to Hugging Face for providing tools essential for this project.  

**Team Members**: 
-**Rane Dy(Project Lead)**
-**Zhuohan Zhang**
