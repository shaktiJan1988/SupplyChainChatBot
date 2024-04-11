import pandas as pd
import streamlit as st
from streamlit_chat import message
import openai
from openai import OpenAI
import json
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
API_KEY = os.getenv("OPENAI_API_KEY")
# Command stored in a string variable
command = "df.head()"
if 'generated' not in st.session_state:
        st.session_state['generated'] = []

if 'past' not in st.session_state:
        st.session_state['past'] = []


# Execute the command



def load_data():
    df = pd.DataFrame( {
    'ProductId': [101, 102, 103, 104, 105],
    'Price': [14.99, 23.50, 7.99, 19.99, 2.99],
    'Color': ['Red', 'Yellow', 'Green', 'Black', 'Yellow']})

    return df

def generate_prompt(query):
        prompt = (""" For the following query, if it requires generate a sql query, reply as follows:
            {"pandas_query":"pandas_query" }
            Example:
            {"answer": Products_df.head()"}
             The query should consider to ignore case-sensitive    
             
            Return all output as a json.

            The Products_df dataframe has 3 colums with names roductId,Price and Color

            Below is the query.
            Query:
            """
            + query )
        return prompt

def get_query_from_llm(prompt, openai_api_key):
    # Set up OpenAI API key
    #openai.api_key = openai_api_key
    client = OpenAI(api_key=openai_api_key)

    # Send the text to the LLM
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": str(prompt),
        }
    ],
    model="gpt-3.5-turbo",
)

    # Extract the SQL query from the response
    pandas_query = chat_completion.choices[0].message.content
    
    return pandas_query.__str__()

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)


st.title("👨‍💻 Chat with your Data")
query = st.text_area("Insert your query")

if st.button("Submit Query", type="primary"):
  Products_df=load_data()
  prompt=generate_prompt(query)
  response=get_query_from_llm(prompt,API_KEY)
  decoded_response = decode_response(response)
  print(decoded_response['pandas_query'])
  executed_command = eval(decoded_response['pandas_query'])
  print(executed_command)
  st.write(executed_command)
  st.session_state.past.append(query)
  st.session_state.generated.append(executed_command)
    

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    
