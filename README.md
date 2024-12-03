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
