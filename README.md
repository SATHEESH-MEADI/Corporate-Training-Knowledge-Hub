<!--
# KHIA-Tech



Using this command log in locally and set you access token so that you can directly access the model from hugging face with out locally loading it 

huggingface-cli login

This command sends the requirements in the project to requirements.txt and then you can easily download instead of pushing the whole model and installing one by one.


pip freeze > requirements.txt

## steps to run application
1. Install python version 3.10
1. Install ollama 
1. open command promt(cmd) and run command 'ollama run llama3.2'
1. navigate to project directory in cmd 
1. Run command 'pip install -r requirements.txt'
1. Once all the required packages are installed run 'streamlit run main.py'


-->




# Corporate Training Knowledge Hub

Welcome to the **Corporate Training Knowledge Hub**, an innovative, data-driven solution designed to transform how organizations interact with training materials. This powerful web application provides a range of advanced features to help corporate trainers, employees, and learners efficiently navigate, analyze, and utilize corporate training content. Whether you're looking to summarize dense documents, generate quizzes, or extract insights from your training materials, this hub is your one-stop solution.

## Key Features

### 1. **Seamless File Upload**
   - **Multi-format Support**: Upload corporate documents in multiple formats including PDF, PPTX, TXT, and XLSX.
   - **Automatic Content Extraction**: The app processes and extracts relevant content, enabling easy access to key training information.

### 2. **Document Summarization**
   - **AI-powered Summaries**: Generate concise, meaningful summaries for long and complex documents using the cutting-edge LLaMA 3.2 model.
   - **Time-saving**: Get a quick overview of training materials to save time and focus on essential learning points.

###3. **Interactive Q&A**
- **AI-driven Q&A**: Ask detailed questions about your documents and get instant, relevant answers. This functionality is powered by Retrieval-Augmented Generation (RAG), which combines the power of document retrieval with the conversational capabilities of the LLaMA 3.2 model. RAG allows the system to search for relevant information from your uploaded documents and use that information to generate contextually rich, precise answers.
- **Contextual Awareness**: The app intelligently references the actual content of the documents uploaded by users. By utilizing the RAG methodology, the app retrieves the most relevant sections of the document based on the user's query. This ensures that responses are not only accurate but also tailored to the specific context of the question, leading to more natural and informative interactions.
- **Dynamic Search & Response Generation:** When you ask a question, the system first searches the uploaded documents for the most relevant content using a vector store (powered by FAISS). Once the relevant information is retrieved, the LLaMA 3.2 model processes it to generate an answer that directly addresses your query, ensuring that the response is grounded in the document's actual content.
- **Enhanced Conversational Flow:** The system maintains a conversational memory, tracking the context of previous questions and answers to provide responses that evolve in a more human-like, context-aware manner. This memory feature, combined with document retrieval, ensures that the answers are coherent over the course of the interaction.








### 4. **Quiz Generation**
   - **Automated Quiz Creation**: Automatically generate multiple-choice quizzes based on the content of any uploaded document.
   - **Instant Feedback**: Users can submit answers and receive real-time feedback, making it ideal for self-assessment or training evaluation.

### 5. **Word Cloud Visualization**
   - **Visual Insights**: Generate dynamic word clouds that highlight key terms and concepts across your uploaded documents.
   - **Data Visualization**: Quickly grasp the main themes and concepts by visualizing the most frequent words in your content.

### 6. **Document Comparison**
   - **Side-by-side Document Comparison**: Compare two documents to identify similarities, differences, and unique insights.
   - **Interactive Diff View**: View a detailed comparison with an HTML-based diff viewer, making it easy to spot changes and discrepancies.

### 7. **Entity Highlighting**
   - **Named Entity Recognition (NER)**: Extract important entities like persons, organizations, locations, and dates from your documents.
   - **Concise Descriptions**: Receive clear, AI-generated descriptions of key entities to enhance understanding and focus on the most important aspects of your training material.

### 8. **Highlights & Insights**
   - **Refined Contextual Insights**: Extract highlights from documents with context-based descriptions for entities, providing valuable summaries of key information.

## Benefits

- **Efficiency**: Quickly navigate vast amounts of training data and focus on what's important.
- **Improved Learning**: The app enhances the learning process by summarizing content, answering questions, and reinforcing key concepts.
- **Data-Driven Insights**: Use AI-powered analytics to make more informed decisions about training materials and their impact on learning.
- **Customization**: Generate quizzes and insights tailored to your specific training materials, ensuring the relevance and effectiveness of your learning tools.

## Installation Guide

### 1. Clone the Repository

To get started with the **Corporate Training Knowledge Hub**, clone the repository to your local machine:

```bash
git clone https://github.com/SATHEESH-MEADI/Corporate-Training-Knowledge-Hub.git
cd corporate-training-knowledge-hub
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install all required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

Launch the app with Streamlit:

```bash
streamlit run main.py
```

The app will start locally at `http://localhost:8501`, ready for you to explore and engage with your documents!

## Technologies Behind the Hub

- **Streamlit**: The framework for building the interactive web interface.
- **LangChain**: A robust framework for developing advanced language processing applications, used for document summarization, Q&A, and entity extraction.
- **Ollama (LLaMA 3.2)**: The foundation model driving our intelligent Q&A and summarization capabilities, enabling rich conversational interactions and high-quality content generation.
- **Hugging Face Transformers**: For natural language processing tasks such as Named Entity Recognition (NER) and text generation.
- **FAISS**: A powerful tool for document retrieval, enabling efficient similarity search across large corpora.
- **Retrieval-Augmented Generation (RAG):** A powerful approach combining document retrieval with generative language models. RAG enhances the interactive Q&A by first retrieving the most relevant content from the uploaded documents using FAISS and then generating a detailed and contextually accurate answer using the Ollama (LLaMA 3.2) model. This enables precise, context-aware responses to user queries, leveraging the strengths of both retrieval and generation for better user interaction.

## Contributing

We welcome contributions to improve and expand the **Corporate Training Knowledge Hub**. Feel free to submit bug reports, feature requests, or pull requests. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## Contact & Support

For any inquiries or support, feel free to reach out to us at smeadi1@umbc.edu. If you find any issues or have questions about how to use the app, we are here to help!


## Acknowledgments

- **Ollama (LLaMA 3.2)**: For powering the advanced natural language capabilities of the app.
- **Streamlit**: For creating an intuitive and easy-to-use web interface for data science apps.
- **LangChain**: For providing the tools to chain language models and build complex document processing workflows.



## 👨‍💻 Author  

**Satheesh Meadi**  
Master's Student in Data Science | NLP Enthusiast  
📧 Email: smeadi1@umbc.edu  
🌐 GitHub: [GitHub](https://github.com/SATHEESH-MEADI)  
📚 LinkedIn: [Satheesh Meadi](https://www.linkedin.com/in/satheesh-meadi/)
