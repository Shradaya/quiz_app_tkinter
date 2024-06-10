import random

def get_questions():
    questions = [
        {
            "question": "Which of the following is a common file format used for storing tabular data?",
            "options": ["CSV", "MP4", "JPG", "PDF"],
            "answer": "CSV",
            "explanation": "CSV (Comma Separated Values) is a widely used file format for storing tabular data because it is simple, human-readable, and supported by many applications. It represents data in a plain text format with values separated by commas."
        },
        {
            "question": "What does the acronym API stand for in the context of data engineering?",
            "options": ["Automated Programming Interface", "Application Program Interface", "Advanced Programming Instruction", "Association of Programmed Interaction"],
            "answer": "Application Program Interface",
            "explanation": "In the context of data engineering, an API (Application Programming Interface) defines methods for interacting with software applications or systems to access and manipulate data."
        },
        {
            "question": "Which of the following is NOT a NoSQL database?",
            "options": ["MongoDB", "Cassandra", "MySQL", "Redis"],
            "answer": "MySQL",
            "explanation": "MySQL is a relational database management system (RDBMS), not a NoSQL database. NoSQL databases are designed for unstructured or semi-structured data and do not require fixed table schemas like relational databases."
        },
        {
            "question": "In data engineering, what is the purpose of a schema?",
            "options": ["To represent the structure of data", "To analyze data trends", "To optimize database performance", "To visualize data relationships"],
            "answer": "To represent the structure of data",
            "explanation": "A schema in data engineering defines the structure of data, including the types of data that can be stored, the format of the data, and any constraints on the data. It provides a blueprint for organizing and interpreting data within a database or data warehouse."
        },
        {
            "question": "Which of the following is a key-value pair storage system used in distributed data processing?",
            "options": ["Hadoop", "Cassandra", "Kafka", "Spark"],
            "answer": "Cassandra",
            "explanation": "Apache Cassandra is a distributed NoSQL database management system designed to handle large amounts of data across many commodity servers, providing high availability with no single point of failure. It stores data in a distributed, decentralized architecture using a key-value pair storage model."
        },
        {
            "question": "What does the acronym SQL stand for?",
            "options": ["Standard Query Language", "Structured Query Language", "System Query Language", "Secure Query Language"],
            "answer": "Structured Query Language",
            "explanation": "SQL (Structured Query Language) is a domain-specific language used in programming and designed for managing data held in a relational database management system (RDBMS), or for stream processing in a relational data stream management system (RDSMS)."
        },
        {
            "question": "Which of the following is NOT a common method for data storage in cloud computing?",
            "options": ["Object Storage", "Block Storage", "Relational Storage", "File Storage"],
            "answer": "Relational Storage",
            "explanation": "Relational Storage is not a common method for data storage in cloud computing. Cloud storage methods typically include object storage, block storage, and file storage. Relational databases are commonly used for structured data storage but are not native to cloud computing environments."
        },
        {
            "question": "What is the primary role of a data engineer in the data pipeline process?",
            "options": ["Data analysis", "Model training", "Data ingestion and transformation", "Data visualization"],
            "answer": "Data ingestion and transformation",
            "explanation": "The primary role of a data engineer in the data pipeline process is to handle data ingestion and transformation. This involves collecting raw data from various sources, cleaning and preprocessing it, and transforming it into a format suitable for analysis or storage in a data warehouse or database."
        },
        {
            "question": "Which of the following is a common open-source stream processing platform?",
            "options": ["Apache Kafka", "Apache Spark", "Apache Hadoop", "Apache Cassandra"],
            "answer": "Apache Kafka",
            "explanation": "Apache Kafka is a common open-source stream processing platform used for building real-time data pipelines and streaming applications. It provides high-throughput, fault-tolerant messaging, and is designed to handle large volumes of data streams across distributed systems."
        },
        {
            "question": "What is the purpose of data partitioning in distributed data processing?",
            "options": ["To increase data redundancy", "To optimize data storage", "To improve query performance", "To enhance data security"],
            "answer": "To improve query performance",
            "explanation": "Data partitioning in distributed data processing involves dividing large datasets into smaller, more manageable partitions that can be processed independently across multiple nodes or servers. This improves query performance by distributing the workload and allowing parallel processing of data."
        },
        {
            "question": "What is the CAP theorem in distributed systems?",
            "options": ["Consistency, Availability, Partition Tolerance", "Consistency, Atomicity, Persistence", "Concurrency, Atomicity, Persistence", "Convergence, Availability, Persistence"],
            "answer": "Consistency, Availability, Partition Tolerance",
            "explanation": "The CAP theorem, also known as Brewer's theorem, states that in a distributed system, it is impossible to simultaneously guarantee all three of the following properties: consistency, availability, and partition tolerance. Distributed systems must sacrifice one of these properties to ensure the other two."
        },
        {
            "question": "Which of the following is a common tool used for business intelligence?",
            "options": ["Docker", "Tableau", "Kubernetes", "Jenkins"],
            "answer": "Tableau",
            "explanation": "Tableau is a common tool used for business intelligence and data visualization. It allows users to create interactive and shareable dashboards, reports, and visualizations from various data sources, enabling data-driven decision-making and analysis."
        },
        {
            "question": "What is the purpose of data normalization in databases?",
            "options": ["To reduce redundancy and improve data integrity", "To encrypt sensitive data", "To optimize data storage", "To convert data into a standard format"],
            "answer": "To reduce redundancy and improve data integrity",
            "explanation": "Data normalization in databases is the process of organizing data to minimize redundancy and dependency by dividing large tables into smaller, more manageable tables and defining relationships between them. This improves data integrity and reduces the likelihood of anomalies or inconsistencies."
        },
        {
            "question": "Which of the following is NOT a type of database index?",
            "options": ["B-tree", "Hash", "Radix", "Binary"],
            "answer": "Binary",
            "explanation": "Binary is not a type of database index. Common types of database indexes include B-tree, hash, and radix indexes, which are used to improve the efficiency of data retrieval by providing fast access to specific data records or subsets of data."
        },
        {
            "question": "What is the difference between a primary key and a foreign key in a relational database?",
            "options": ["A primary key uniquely identifies a record in a table, while a foreign key refers to a primary key in another table", "A primary key is used for indexing, while a foreign key is used for encryption", "A primary key is used for encryption, while a foreign key is used for indexing", "There is no difference between them"],
            "answer": "A primary key uniquely identifies a record in a table, while a foreign key refers to a primary key in another table",
            "explanation": "In a relational database, a primary key is a column or a set of columns that uniquely identifies each row in a table. It ensures data integrity and enforces entity integrity constraints. A foreign key, on the other hand, is a column or a set of columns in a table that establishes a relationship with a primary key or a unique key in another table. It enforces referential integrity constraints and maintains data consistency between related tables."
        },
        {
            "question": "Which of the following is NOT a common data cleaning task?",
            "options": ["Removing duplicates", "Handling missing values", "Standardizing data formats", "Creating indexes"],
            "answer": "Creating indexes",
            "explanation": "Creating indexes is not a common data cleaning task. Data cleaning tasks typically involve removing duplicates, handling missing values, standardizing data formats, and resolving inconsistencies to ensure data quality and accuracy."
        },
        {
            "question": "What is the purpose of a data lake?",
            "options": ["To store structured data in a hierarchical format", "To store unstructured and semi-structured data at scale", "To perform real-time data processing", "To visualize data relationships"],
            "answer": "To store unstructured and semi-structured data at scale",
            "explanation": "A data lake is a centralized repository that allows you to store all your structured, unstructured, and semi-structured data at any scale. It enables you to break down data silos and analyze data from multiple sources, providing a unified view of your data for analysis and insights."
        },
        {
            "question": "Which of the following is a popular query language for NoSQL databases?",
            "options": ["MongoDB Query Language (MQL)", "Cassandra Query Language (CQL)", "SQL", "A and B"],
            "answer": "A and B",
            "explanation": "Both MongoDB and Cassandra have their own query languages. MongoDB Query Language (MQL) is used for querying data in MongoDB, while Cassandra Query Language (CQL) is used for querying data in Cassandra. SQL is commonly associated with relational databases."
        },
        {
            "question": "What is the purpose of data replication in distributed databases?",
            "options": ["To increase data redundancy", "To decrease data consistency", "To optimize data storage", "To reduce data integrity"],
            "answer": "To increase data redundancy",
            "explanation": "Data replication in distributed databases involves creating and maintaining multiple copies of data across different nodes or servers within a network. It helps improve fault tolerance, increase data availability, and ensure data durability by providing redundancy and backups of data."
        },
        {
            "question": "Which of the following is a characteristic of columnar databases?",
            "options": ["Optimized for transactions", "Store data in rows", "Optimized for analytical queries", "Use a key-value pair storage model"],
            "answer": "Optimized for analytical queries",
            "explanation": "Columnar databases are optimized for analytical queries and data warehousing applications. Unlike row-based databases, which store data in rows, columnar databases store data in columns. This allows for efficient querying and processing of large volumes of data for analytical purposes, such as business intelligence and data analytics."
        },
        {
            "question": "Which of the following is NOT a supervised learning algorithm?",
            "options": ["K-means", "Decision Trees", "Support Vector Machines", "Linear Regression"],
            "answer": "K-means",
            "explanation": "K-means is a clustering algorithm used for unsupervised learning, not a supervised learning algorithm. Supervised learning algorithms require labeled training data, while unsupervised learning algorithms do not."
        },
        {
            "question": "What does the acronym SVM stand for in machine learning?",
            "options": ["Supervised Vector Machine", "Simple Vector Machine", "Support Vector Machine", "Singular Value Machine"],
            "answer": "Support Vector Machine",
            "explanation": "SVM stands for Support Vector Machine, which is a supervised learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates different classes or groups in the input data space."
        },
        {
            "question": "Which evaluation metric is commonly used for imbalanced classification problems?",
            "options": ["Accuracy", "Precision", "F1 Score", "Mean Squared Error"],
            "answer": "F1 Score",
            "explanation": "The F1 Score is commonly used for imbalanced classification problems because it balances both precision and recall, making it suitable for situations where classes are unevenly distributed. It provides a harmonic mean of precision and recall, giving equal weight to false positives and false negatives."
        },
        {
            "question": "What is the purpose of feature scaling in machine learning?",
            "options": ["To remove outliers from the data", "To convert categorical features into numerical ones", "To standardize the range of features", "To encode text data"],
            "answer": "To standardize the range of features",
            "explanation": "Feature scaling in machine learning is the process of standardizing or normalizing the range of independent variables or features in the dataset. It ensures that features with different scales and units contribute equally to the model training process, preventing features with larger scales from dominating the learning algorithm."
        },
        {
            "question": "What is the term used to describe the phenomenon where a model performs well on the training data but poorly on unseen data?",
            "options": ["Overfitting", "Underfitting", "Bias", "Variance"],
            "answer": "Overfitting",
            "explanation": "Overfitting occurs when a machine learning model learns the training data too well, capturing noise or random fluctuations in the data rather than the underlying patterns. As a result, the model performs well on the training data but poorly on unseen data, failing to generalize to new or unseen examples."
        },
        {
            "question": "Which of the following is NOT a dimensionality reduction technique?",
            "options": ["Principal Component Analysis (PCA)", "Linear Discriminant Analysis (LDA)", "K-means Clustering", "t-Distributed Stochastic Neighbor Embedding (t-SNE)"],
            "answer": "K-means Clustering",
            "explanation": "K-means Clustering is a clustering algorithm used for unsupervised learning, not a dimensionality reduction technique. Dimensionality reduction techniques aim to reduce the number of input variables or features in a dataset while preserving the most important information or patterns."
        },
        {
            "question": "What is the purpose of cross-validation in machine learning?",
            "options": ["To train a model on multiple datasets simultaneously", "To test a model's performance on unseen data", "To select the best hyperparameters for a model", "To assess a model's generalization ability"],
            "answer": "To assess a model's generalization ability",
            "explanation": "Cross-validation in machine learning is a technique used to assess the generalization ability or predictive performance of a model. It involves splitting the dataset into multiple subsets, training the model on a subset of the data, and evaluating its performance on the remaining subsets. This helps estimate how well the model will perform on unseen data."
        },
        {
            "question": "Which of the following activation functions is NOT commonly used in neural networks?",
            "options": ["ReLU (Rectified Linear Activation)", "Tanh (Hyperbolic Tangent)", "Sigmoid", "Softmax"],
            "answer": "Softmax",
            "explanation": "Softmax is an activation function commonly used in the output layer of neural networks for multi-class classification problems. ReLU, Tanh, and Sigmoid are commonly used activation functions in hidden layers of neural networks for introducing non-linearity and enabling the model to learn complex patterns."
        },
        {
            "question": "In machine learning, what does the term 'bias' refer to?",
            "options": ["The difference between predicted and actual values", "The flexibility of a model to fit complex patterns", "The error due to overly simplistic assumptions in the learning algorithm", "The sensitivity of a model to variations in the input data"],
            "answer": "The error due to overly simplistic assumptions in the learning algorithm",
            "explanation": "In machine learning, bias refers to the error introduced by overly simplistic assumptions in the learning algorithm. High bias can lead to underfitting, where the model is too simple to capture the underlying structure of the data, resulting in poor predictive performance."
        },
        {
            "question": "Which of the following techniques is used to handle missing data in a dataset?",
            "options": ["Feature engineering", "Feature scaling", "Imputation", "Normalization"],
            "answer": "Imputation",
            "explanation": "Imputation is a technique used to handle missing data in a dataset by replacing missing values with estimated values based on the available data. Common imputation methods include mean imputation, median imputation, and mode imputation, which preserve the overall distribution of the data."
        },
        {
            "question": "What is the purpose of regularization in machine learning?",
            "options": ["To increase model complexity", "To reduce model complexity", "To introduce noise into the training data", "To prevent overfitting"],
            "answer": "To prevent overfitting",
            "explanation": "Regularization in machine learning is a technique used to prevent overfitting by adding a penalty term to the loss function that penalizes overly complex models. It discourages large weights or coefficients in the model, encouraging simpler models that generalize better to unseen data."
        },
        {
            "question": "Which of the following algorithms is commonly used for anomaly detection?",
            "options": ["Naive Bayes", "K-means", "Isolation Forest", "Gradient Boosting"],
            "answer": "Isolation Forest",
            "explanation": "Isolation Forest is a machine learning algorithm commonly used for anomaly detection. It works by isolating anomalies or outliers in the dataset by randomly partitioning the data into subsets and identifying anomalies based on their isolation from the rest of the data points."
        },
        {
            "question": "Which of the following is NOT a hyperparameter for tuning a decision tree model?",
            "options": ["Maximum Depth", "Learning Rate", "Minimum Samples Split", "Criterion"],
            "answer": "Learning Rate",
            "explanation": "Learning Rate is not a hyperparameter for tuning a decision tree model. Hyperparameters for tuning decision trees typically include maximum depth, minimum samples split, criterion (e.g., Gini impurity or entropy), and minimum samples leaf."
        },
        {
            "question": "What does the term 'ensemble learning' refer to in machine learning?",
            "options": ["The process of training multiple models simultaneously", "The process of combining the predictions of multiple models to improve performance", "The process of evaluating a model's performance using cross-validation", "The process of selecting the best features for a model"],
            "answer": "The process of combining the predictions of multiple models to improve performance",
            "explanation": "Ensemble learning in machine learning refers to the process of combining the predictions of multiple models to improve overall performance. It leverages the diversity of individual models to reduce errors and increase predictive accuracy, often resulting in better generalization and robustness."
        },
        {
            "question": "Which of the following techniques is used to handle class imbalance in classification tasks?",
            "options": ["Oversampling", "Undersampling", "SMOTE (Synthetic Minority Over-sampling Technique)", "All of the above"],
            "answer": "All of the above",
            "explanation": "All of the mentioned techniques - oversampling, undersampling, and SMOTE (Synthetic Minority Over-sampling Technique) - are used to handle class imbalance in classification tasks. These techniques aim to balance the distribution of classes in the training data, improving the performance of machine learning models on imbalanced datasets."
        },
        {
            "question": "In machine learning, what does the term 'bias' refer to?",
            "options": ["The error due to overly simple assumptions in the learning algorithm", "The error due to noise in the training data", "The error due to variability in the dataset", "The error due to high complexity of the model"],
            "answer": "The error due to overly simple assumptions in the learning algorithm",
            "explanation": "In machine learning, bias refers to the error introduced by overly simple assumptions in the learning algorithm. High bias can lead to underfitting, where the model is too simple to capture the underlying structure of the data, resulting in poor predictive performance."
        },
        {
            "question": "Which of the following is NOT a common evaluation metric for regression models?",
            "options": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R-squared (Coefficient of Determination)", "Area Under the ROC Curve (AUC-ROC)"],
            "answer": "Area Under the ROC Curve (AUC-ROC)",
            "explanation": "Area Under the ROC Curve (AUC-ROC) is a common evaluation metric for classification models, not regression models. Common evaluation metrics for regression models include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (Coefficient of Determination)."
        },
        {
            "question": "Which of the following techniques is commonly used for dimensionality reduction?",
            "options": ["Principal Component Analysis (PCA)", "K-nearest Neighbors (KNN)", "Logistic Regression", "Random Forest"],
            "answer": "Principal Component Analysis (PCA)",
            "explanation": "Principal Component Analysis (PCA) is a common technique used for dimensionality reduction in machine learning and data analysis. It transforms high-dimensional data into a lower-dimensional space while preserving as much of the variability in the data as possible, making it easier to visualize and analyze."
        },
        {
            "question": "What is the purpose of one-hot encoding in machine learning?",
            "options": ["To transform categorical variables into numerical ones", "To normalize the range of features", "To standardize the scale of features", "To remove outliers from the data"],
            "answer": "To transform categorical variables into numerical ones",
            "explanation": "One-hot encoding is a technique used to transform categorical variables into numerical ones in machine learning. It creates binary dummy variables for each category or level of a categorical variable, allowing the model to interpret and process categorical data as numerical features."
        },
        {
            "question": "Which of the following algorithms is NOT a type of ensemble learning?",
            "options": ["Random Forest", "Gradient Boosting", "K-nearest Neighbors (KNN)", "AdaBoost"],
            "answer": "K-nearest Neighbors (KNN)",
            "explanation": "K-nearest Neighbors (KNN) is not a type of ensemble learning algorithm. Random Forest, Gradient Boosting, and AdaBoost are examples of ensemble learning algorithms that combine the predictions of multiple base learners to improve overall performance."
        }
    ]
    random.shuffle(questions)
    return questions if questions else []
