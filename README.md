# Multilingual News Framing Analysis

The repository contains the demo code for news framing analysis based on ChatGPT and LangChain.

The code applies the following tasks:

1- Idenitifies the article's language. This step is important for handling multilingualism.

2- It identifies the check-worthy entities that shapes the news narrative.

3- Based on detected entities, it identifies the article's stance towards entities and selects sentences from the article as evidences of its rationales.

The steps for running demo:
- Create a virtual environment
```shell
virtualenv -p python3.10 venv
```
- Install requirements
```shell
pip install -r requirements.txt
```
- Create a .env file
```text
OPENAI_API_KEY=API_KEY
CHATGPT_MODEL=gpt-3.5-turbo-1106
OPENAI_ORG=ORG_KEY
```
- Run the Stremlit demo
```shell
streamlit run app.py
```
## Acknowledgement
The work is supported by KID2.
