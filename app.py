import pandas as pd
import streamlit as st
from streamlit_chat import message
import openai
from openai import OpenAI
import json
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import pandas as pd
import joblib
from PIL import Image
import os
#API_KEY = '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'
API_KEY = os.getenv("OPENAI_API_KEY")
# Command stored in a string variable
command = "df.head()"

if 'generated' not in st.session_state:
        st.session_state['generated'] = []

if 'past' not in st.session_state:
        st.session_state['past'] = []


# Execute the command



def load_data():
    #df = pd.DataFrame( {
    #'ProductId': [101, 102, 103, 104, 105,106,107,108,109,110],
    #'Price': [14.99, 23.50, 7.99, 19.99, 2.99,14.99, 23.50, 7.99, 19.99, 2.99],
    #'Color': ['Red', 'Yellow', 'Green', 'Black', 'Yellow','Red', 'Yellow', 'Green', 'Black', 'Yellow']})
    df=pd.read_csv('Dataset/supply_chain_data.csv')
    return df

def generate_prompt(query):
        prompt = (""" For the following query, if it requires generate a  query to plot, reply as follows:
                   {"pandas_plot": {"layers": [{"query": "sns.barplot(x='SKU', y='Price',label='Price vs SKU',data=Products_df)"},{"query": "sns.lineplot(x='SKU', y='Price', label='Price vs SKU',data=Products_df)"},{"query": "sns.histplot(data = Products_df, x = 'Price', kde = True, hue = 'Product type')"},{"query": "plt.pie(Products_df.groupby('Product type')['Price'].sum(), labels=df.groupby('Product type')['Price'].sum().index, autopct='%1.1f%%', startangle=90)"},{"query": "sns.relplot(y ='Price', x ='SKU', hue ='Product type', style ='Transportation modes',  data = Products_df).set_xticklabels(rotation=90)"},{"query": "sns.catplot(x='SKU',  y='Price', hue='Product type', col='Transportation modes',kind='bar', data=Products_df)"},{"query": "sns.heatmap(Products_df.corr(), annot=True, annot_kws={'size': 5})"}]}}
            elseif asked to generate a query reply with string as follows   
            {"pandas_query":"Products_df.head()" }
             elseif asked to predict the price or suggest i  reply with string as follows       
            {"predict_price":{"Product type":["haircare"],"Location":["Chennai"],"Order quantities":[100],"Manufacturing costs":[50],"Transportation modes":["Road"],"Shipping costs":[10]} }     
             elseif not able to generate the correct response reply as follows
                 {"Invalid Response":"The Query is Invalid" }  
                  The query should consider to ignore case-sensitive    
            The Products_df dataframe has  colums with names {Product type,SKU,Price,Availability,Number of products sold,Revenue generated,Customer demographics,Stock levels,Lead times,Order quantities,Shipping times,Shipping carriers,Shipping costs,Supplier name,Location,Lead time,Production volumes,Manufacturing lead time,Manufacturing costs,Inspection results,Defect rates,Transportation modes,Routes,Costs}

            Below is the query.
            Query
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


st.title("👨‍💻       Supply Chain Analytics")
image_path = os.path.join(os.path.dirname(__file__), 'AI_Image.jpg')
#image = Image.open(image_path)
#new_image = image.resize((200, 100))
#st.image(image, use_column_width=True)
st.subheader("ChatBot")
query = st.text_area("Insert your query here")
if st.button("Submit Query", type="primary"):
  Products_df=load_data()
  prompt=generate_prompt(query)
  response=get_query_from_llm(prompt,API_KEY)
  print(response)
  #decoded_response=response
  decoded_response = decode_response(response)
  #decoded_response={"pandas_plot":{"layers": [{"type": "line","data": {"x": Products_df['ProductId'], "y": Products_df['Price']},"style": {"color": "blue", "label": "Price"}}]}}
  if 'Invalid Response' in decoded_response:
     print(decoded_response['Invalid Response'])
     executed_command = (decoded_response['Invalid Response'])
     print(executed_command)
   # Token_detail="Input Tokens:-"+decoded_response['Input Token length']+'  '+"OutputTokens:-"+decoded_response['Output Token']
  #  st.write(Token_detail)
     st.write(executed_command)
     st.session_state.past.append(query)
     st.session_state.generated.append(executed_command)

     if st.session_state['generated']:
      for i in range(len(st.session_state['generated'])-1, -1, -1):
       message(str(st.session_state['generated'][i]),key=str(i))
       message(st.session_state['past'][i], is_user=True,key=str(i) + '_user')
  elif 'pandas_query' in decoded_response:
     print(decoded_response['pandas_query'])
     executed_command = eval(decoded_response['pandas_query'])
     print(executed_command)
    # Token_detail="Input Tokens:-"+decoded_response['Input Token length']+'  '+"OutputTokens:-"+decoded_response['Output Token']
   #  st.write(Token_detail)
     st.write(executed_command)
     st.session_state.past.append(query)
     st.session_state.generated.append(executed_command)

     if st.session_state['generated']:
      for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(str(st.session_state['generated'][i]),key=str(i))
        message(st.session_state['past'][i], is_user=True,key=str(i) + '_user')
  elif 'pandas_plot' in decoded_response:
      print(decoded_response['pandas_plot'])
      graph_data = decoded_response['pandas_plot']
     #fig, ax = plt.subplots()
     # Add layers from the JSON data

      for layer in graph_data['layers']:
          eval(layer['query'])
      plt.legend()
      plt.show()
      st.pyplot(plt)
  elif 'predict_price' in decoded_response:
       required_keys = ["Product type", "Location", "Order quantities", "Manufacturing costs", "Transportation modes"]
       if all(key in decoded_response['predict_price'] for key in required_keys):
                price_pred_model = joblib.load(os.path.join(os.path.dirname(__file__), 'product_price_model.pkl'))
                le_product_type = joblib.load(os.path.join(os.path.dirname(__file__), 'product_type_encoder.pkl'))
                le_location = joblib.load(os.path.join(os.path.dirname(__file__), 'location_encoder.pkl'))
                le_transportation_mode = joblib.load(os.path.join(os.path.dirname(__file__), 'transportation_mode_encoder.pkl'))
                features= joblib.load(os.path.join(os.path.dirname(__file__), 'price_pred_features.pkl'))
# Create a dictionary with the input parameters
                input_data = {'Product type': (decoded_response['predict_price']['Product type']) ,'Location': (decoded_response['predict_price']['Location']),'Order quantities': (decoded_response['predict_price']['Order quantities']),'Manufacturing costs': (decoded_response['predict_price']['Manufacturing costs']),'Transportation modes': (decoded_response['predict_price']['Transportation modes']),'Shipping costs': (decoded_response['predict_price']['Shipping costs'])}

                print(input_data)
                input_df = pd.DataFrame(input_data)
                input_df['Product_type_encoded'] = le_product_type.transform(input_df['Product type'])
                input_df['Location_encoded'] = le_location.transform(input_df['Location'])
                input_df['Transportation_mode_encoded'] = le_transportation_mode.transform(input_df['Transportation modes'])
                pred_inp=input_df[features]
                predicted_price = price_pred_model.predict(pred_inp)
                st.write(predicted_price)
                st.session_state.past.append(query)
                st.session_state.generated.append(predicted_price)
                if st.session_state['generated']:
                    for i in range(len(st.session_state['generated'])-1, -1, -1):
                         message(str(st.session_state['generated'][i]),key=str(i))
                         message(st.session_state['past'][i], is_user=True,key=str(i) + '_user')
       else:
                st.write("Invalid parameters")
                st.session_state.past.append(query)
                st.session_state.generated.append("Invalid parameters")

                if st.session_state['generated']:
                   for i in range(len(st.session_state['generated'])-1, -1, -1):
                      message(str(st.session_state['generated'][i]),key=str(i))
                      message(st.session_state['past'][i], is_user=True,key=str(i) + '_user')
# Show legend


# Display the figure in Streamlit

    #  executed_command = eval(decoded_response['pandas_plot'])
    #  print(executed_command)
    #  st.write(executed_command)
    #  st.session_state.past.append(query)
    #  st.session_state.generated.append(executed_command)

  

      #  message(st.session_state["generated"][i], key=str(i))
     #  message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

#if __name__ =='__main__':
#   st.set_option('sever.enableCORS,TRUE')

