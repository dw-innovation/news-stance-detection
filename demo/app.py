import streamlit as st
import os
import json
from loguru import logger
from langchain.globals import set_llm_cache
from typing import List, Optional, Any, Dict
from langchain.cache import SQLiteCache
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


def read_task_prompt(fname):
    with open(fname, 'r') as f:
        return f.read()


class Stance(BaseModel):
    entity: str
    label: str
    evidences: Optional[str] = []


class Stances(BaseModel):
    stances: List[Stance]


parser = PydanticOutputParser(pydantic_object=Stances)

prompt_task_1 = ChatPromptTemplate.from_template(read_task_prompt(fname='../prompts/task_1.txt'))
prompt_task_2 = ChatPromptTemplate.from_template(read_task_prompt(fname='../prompts/task_2.txt'))
prompt_task_3 = ChatPromptTemplate.from_template(read_task_prompt(fname='../prompts/task_3.txt'), partial_variables={
    "format_instructions": parser.get_format_instructions()})

model = ChatOpenAI(temperature=0,
                   openai_api_key=os.getenv('OPENAI_API_KEY'),
                   openai_organization=os.getenv('OPENAI_ORG'),
                   model_name=os.getenv('CHATGPT_MODEL'),
                   verbose=True)

model_parser = model | StrOutputParser()

article_input = {"article": RunnablePassthrough()}

chain_task_1 = prompt_task_1 | model_parser

chain_task_2 = {"article": article_input, "language": chain_task_1} | prompt_task_2 | model | StrOutputParser()

chain_task_3 = {"article": article_input, "language": chain_task_1,
                "entities": chain_task_2} | prompt_task_3 | model | StrOutputParser()


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        logger.info(f"Prompt:\n{formatted_prompts}")


# Function to highlight sentences based on model output
def highlight_sentences(text, model_output):
    highlighted_text = ""
    sentences = text.split('. ')  # Split text into sentences (assuming period-space is the sentence separator)

    for i, sentence in enumerate(sentences):
        if i in model_output:
            highlighted_text += f"<mark>{sentence}</mark> "
        else:
            highlighted_text += f"{sentence}. "

    return highlighted_text


# Streamlit app
def main():
    st.title("News Stance Detection")

    # Input text
    text = st.text_area("Paste the news article:", height=500)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def set_clicked():
        st.session_state.clicked = True

    # Highlight button
    st.button("Detect Stances", on_click=set_clicked)

    if st.session_state.clicked:
        # Highlight sentences based on model output
        model_output = chain_task_3.invoke({"article": text}, config={"callbacks": [CustomHandler()]})
        model_output = json.loads(model_output)
        st.subheader("Model Output")
        stances = model_output["stances"]

        highlighted_text = text
        st.markdown("### Detected Entities and The News Stance Towards the Entities")

        for stance in stances:
            entity_name = stance['entity']
            label = stance['label']
            if label == 'favor':
                color = ':green'
            elif label == 'against':
                color = ':red'
            else:
                color = ':gray'
            st.markdown(f"Entity is **{entity_name}** and news stance towards this entity is **{color}[{label}]**")

        st.markdown("### Select entities to higlight the evidence sentences")

        # Categories selection
        entities = list(map(lambda x: x['entity'], stances))
        entities.append('all')

        selected_entities = st.multiselect("Entities:", entities)

        if 'all' in selected_entities:
            selected_entities = entities

        for stance in stances:
            entity = stance['entity']

            if entity not in selected_entities:
                continue

            label = stance['label']
            evidences = stance['evidences']

            if label == 'favor':
                color = ':green'
            elif label == 'against':
                color = ':red'
            else:
                color = ':gray'

            for evidence in evidences:
                highlighted_text = highlighted_text.replace(evidence, f"**{color}[{evidence}]**")

        st.markdown(highlighted_text, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
