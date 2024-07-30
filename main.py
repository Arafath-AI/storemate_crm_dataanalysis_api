from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import FastAPI, Header, HTTPException
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
import os,openai
import jwt
from fastapi import Depends

import requests
import json,time

import google.generativeai as genai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
#from langchain_experimental.agents.agent_toolkits import create_csv_agent


from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index import StorageContext, load_index_from_storage

import base64
def decode_data(encoded_data):
    # Decode and convert back to string
    decoded_bytes = base64.b64decode(encoded_data)
    return str(decoded_bytes.decode('utf-8'))

import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv('OPEN_AI')
gemini_key = os.getenv('GEMINI')


os.environ["OPENAI_API_KEY"] = openai_api_key

try:
    storage_context = StorageContext.from_defaults(persist_dir="llama_index")
    index = load_index_from_storage(storage_context=storage_context)
    print("loaded")
except:     
    documents = SimpleDirectoryReader("data/userguid").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist("llama_index")
    print("index created")

query_engine = index.as_query_engine()
    

import mysql.connector
from decimal import Decimal
from datetime import datetime

# Define the connection parameters
host = "68.183.225.237"
user = "sm_ml"
password = "Fz6/I733"
database = "sm_qa_1"
from real_time_data_getter_class import Data_getter
from datetime import datetime
import pytz
def get_time():
    srilanka_timezone = pytz.timezone('Asia/Colombo')
    current_date_time_srilanka = datetime.now(srilanka_timezone)
    return current_date_time_srilanka.strftime('%Y-%m-%d %I:%M %p')





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#===============================================get_data_analysis section====================================


def get_answer(query,data_path):
    print(data_path)
    genai.configure(api_key=gemini_key)
    
    def upload_to_gemini(path, mime_type=None):
      file = genai.upload_file(path, mime_type=mime_type)
      print(f"Uploaded file '{file.display_name}' as: {file.uri}")
      return file
    
    def wait_for_files_active(files):
      print("Waiting for file processing...")
      for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
          print(".", end="", flush=True)
          time.sleep(10)
          file = genai.get_file(name)
        if file.state.name != "ACTIVE":
          raise Exception(f"File {file.name} failed to process")
      print("...all files ready")
      print()
    
    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash",
      generation_config=generation_config,
      # safety_settings = Adjust safety settings
      # See https://ai.google.dev/gemini-api/docs/safety-settings
    )
    
    # TODO Make these files available on the local file system
    # You may need to update the file paths
    files = [
      upload_to_gemini(data_path, mime_type="text/csv"),
    ]
    
    # Some files have a processing delay. Wait for them to be ready.
    wait_for_files_active(files)
    
    chat_session = model.start_chat(
      history=[
        {
          "role": "user",
          "parts": [
            files[0],
                   ],
        },
      ]
    )
    
    response = chat_session.send_message(f"{query}, give small answers only without code")
    
    return response.text

@app.post("/get_data_analysis_response")
async def get_data_analysis_response(user_name,query,business_id):
    data_path = Data_getter.get_data_from_server(business_id)
    return {"status_code":200,"detail":"valid authorization","message": get_answer(query,data_path)}




#===============================================CRM section===============================================

def save_chat(thread_id,user_name,current_chat):
    # Create a connection to the MySQL server
    connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
    
    print(current_chat)
    current_chat = current_chat.replace("'","")
    print(current_chat)
    if connection.is_connected():
    
    
            # Create a cursor object for executing SQL queries
            cursor = connection.cursor()
            print(thread_id)
            if thread_id!=None:
              query = "UPDATE crm_chat2 SET conversation = CONCAT(conversation, '" + str(current_chat) + "') WHERE tread = "+thread_id
              print("update")
            else:
              query = "INSERT INTO crm_chat2 (user_name,conversation) VALUES('"+user_name+"','"+current_chat+"')"
              print("insert")
            # Execute the SQL query
                
            cursor.execute(query)
            connection.commit()
            print("Connected to MySQL database")
            cursor.close()
            cursor = connection.cursor()
            sql_query = "SELECT tread from crm_chat2 WHERE user_name='"+user_name+"' ORDER BY tread DESC LIMIT 1"

            cursor.execute(sql_query)
            current_thread_id = cursor.fetchone()[0]
            
            connection.close()
            return str(current_thread_id)


"""def generate_data(query,email):
    cookie_path_dir = "./cookies_snapshot"
    sign.saveCookiesToDir(cookie_path_dir)
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
    df = pd.read_csv(email+".csv")
    for resp in chatbot.query(df.to_string()+"    "+query+". dont give method or steps or code only give final results",stream=True):
        yield f"{resp['token']}".encode("utf-8") 
"""
sec_key = "my_sec_for_sm" 

# Dependency for extracting Bearer token
def get_token(authorization: Optional[str] = Header(None)):
    if authorization:
        token_type, _, token = authorization.partition(' ')
        if token_type.lower() != 'bearer':
            return HTTPException(status_code=400, detail="Invalid authentication scheme")
        if not token:
            return HTTPException(status_code=400, detail="Invalid authorization header")
        return token
    else:
        return HTTPException(status_code=401, detail="Authorization header missing")


def validate_token(Bearer_token):
    url = "https://dev2.v6.storemate.parallaxtec.com/api/auth/user"
    
    payload = {}
    headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': 'Bearer '+Bearer_token
    }
    
    return requests.request("GET", url, headers=headers, data=payload).json()  
        


"""          if the question related to greeting like 'hi, hello , how are you , thank you and more' then only answer from your own knowledge like a real human.
          
          otherwise give answer from provided data with make the more details link within the <a href>lank hyper link.
          
          if not find the answer for user question from provided data then only say 'please contact our helpdesk' \n\n

          user question : +user_query)
"""




@app.post("/get_crm_response/")
async def get_crm_response(user_name: str, user_query: str, token: str = Depends(get_token),thread_id:Optional[str] = None):
        token_valid_response = validate_token(token)
        # try:
        #     #jwt.decode(token,sec_key,algorithms=['HS256'])
        #     print(token_valid_response['success'])
        # except:
        #     return {"status_code":401,"detail":"Invalid authorization","message": "This is a secure endpoint"}
        # Create a connection to the MySQL server

        response = query_engine.query("""        
          if you find the answer from provided data then give answer with steps and make the more details link within the <a href>lank hyper link.
          if not find the answer from provided data then say 'please contact our helpdesk' \n\n
          user question : """+user_query)

        print(str(response).lower())
        if "please contact our helpdesk" in str(response).lower() or "please contact" in str(response).lower():
            print("help desk option")

            openai.api_key = os.environ["OPENAI_API_KEY"]

            default = """<br><br>Dear<br>If you have a specific question or need assistance, please feel free to submit a ticket, and our support team will be happy to help you:<br><br>Submit a Ticket:<br>Email: support@storemate.lk<br>Hotline: 0114 226 999<br><br>Thank You """
            messages = [{"role": "user", "content": user_query+".   always give small answers"}]
            response = openai.chat.completions.create(
            
            model="gpt-3.5-turbo",
            
            messages=messages,
            
            temperature=0,
            
            )

            
            current_chat = user_query+"   "+get_time()+"|"+response.choices[0].message.content+default+"   "+get_time()+"|"
            
            return {"status_code":200,"detail":"valid authorization","current_thread_id":save_chat(thread_id,user_name,current_chat),"message": str(response.choices[0].message.content) + default}
            
        result = ""
        for lines in str(response).split("\n"):
            result = result +"<p>"+lines+"</p><br>"

        current_chat = user_query+"   "+get_time()+"|"+result+"   "+get_time()+"|"
            
        return {"status_code":200,"detail":"valid authorization","current_thread_id":save_chat(thread_id,user_name,current_chat),"message": result}
        #return StreamingResponse(generate_data(query,email), media_type="text/plain")



def create_required_format(input_data):
    result = {}
    for username, thread_id, conversation in input_data:
        messages = conversation.split('|')
        # Ensure username is in the result and prepare the structure
        if username not in result:
            result[username] = {"user_name": username, "threads": []}
        thread = {
            "thread_id": thread_id,
            "exchanges": []
        }
        for i in range(0, len(messages), 2):
            try:
                user_message = messages[i]
                bot_message = messages[i + 1]
            except IndexError:
                # In case there's no bot response to the last user message
                bot_message = ""
            thread["exchanges"].append({"user": user_message, "bot": bot_message})
        result[username]["threads"].append(thread)
    
    # Convert to a list of results since multiple usernames might be present
    return list(result.values())

    
@app.post("/get_saved_chat/")
async def get_saved_chat(user_name : str,token: str = Depends(get_token)):
    token_valid_response = validate_token(token)
    # try:
    #         #jwt.decode(token,sec_key,algorithms=['HS256'])
    #         print(token_valid_response['success'])
    # except:
    #         return {"status_code":401,"detail":"Invalid authorization","message": "This is a secure endpoint"}
    # Create a connection to the MySQL server
    connection_load_chat = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )   
    if connection_load_chat.is_connected():
            # Create a cursor object for executing SQL queries
            cursor = connection_load_chat.cursor()
            sql_query = "SELECT * from crm_chat2 WHERE user_name='"+user_name+"'"
            # Execute the SQL query
            cursor.execute(sql_query)
    
            all_result_list = []
            results = cursor.fetchall()
            cursor.close()
            connection_load_chat.close()
            return {"status_code":200,"detail":"valid authorization","message": create_required_format(results)}
    


def create_thread_list(sql_data):
      # Convert the response to the desired format
      threads = {}
      for _, thread_id, content in sql_data:
          parts = content.split('|')
          title = parts[0].strip()
          opened_date = parts[1].split('<')[0].strip()
          threads[thread_id] = {
              "thread_id": thread_id,
              "thread_title": title.split("   ")[0],
              "thread_opened_date": title.split("   ")[-1]
          }
    
      return threads
    
@app.post("/get_saved_threads_of_user/")
async def get_saved_threads_of_user(user_name : str,token: str = Depends(get_token)):
    token_valid_response = validate_token(token)
    # try:
    #         #jwt.decode(token,sec_key,algorithms=['HS256'])
    #         print(token_valid_response['success'])
    # except:
    #         return {"status_code":401,"detail":"Invalid authorization","message": "This is a secure endpoint"}

    # Create a connection to the MySQL server
    connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
    
    if connection.is_connected():
            # Create a cursor object for executing SQL queries
            cursor = connection.cursor()
            sql_query = "SELECT * from crm_chat2 WHERE user_name='"+user_name+"'"
            # Execute the SQL query
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return {"status_code":200,"detail":"valid authorization","message": create_thread_list(results)}

def convert_conversation_to_json(results):
    result_list = results[0][0].split("|")
    json_list = []
    for id in range(0,len(result_list),2):
      try:
        json_list.append({"user_question":result_list[id].split("   ")[0],
                          "qusetion_time":result_list[id].split("   ")[1],
                          "bot_answer":result_list[id+1].split("   ")[0],
                          "answer_time":result_list[id+1].split("   ")[1]})
        
      except:
        pass
    
    return json_list
        
@app.post("/get_conversations_of_one_thread/")
async def get_conversations_of_one_thread(thread_id:str,token: str = Depends(get_token)):
    token_valid_response = validate_token(token)
    print(token_valid_response)
    # try:
    #         #jwt.decode(token,sec_key,algorithms=['HS256'])
    #         print(token_valid_response['success'])
    # except:
    #         return {"status_code":401,"detail":"Invalid authorization","message": "This is a secure endpoint"}

    # Create a connection to the MySQL server
    connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
    
    if connection.is_connected():
            # Create a cursor object for executing SQL queries
            cursor = connection.cursor()
            sql_query = "SELECT conversation from crm_chat2 WHERE tread = "+str(thread_id)
            # Execute the SQL query
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return {"status_code":200,"detail":"valid authorization","message": convert_conversation_to_json(results)}




@app.post("/delete_thread/")
async def delete_thread(thread_id:str,token: str = Depends(get_token)):
    token_valid_response = validate_token(token)
    print(token_valid_response)
    # try:
    #         #jwt.decode(token,sec_key,algorithms=['HS256'])
    #         print(token_valid_response['success'])
    # except:
    #         return {"status_code":401,"detail":"Invalid authorization","message": "This is a secure endpoint"}

    # Create a connection to the MySQL server
    connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
    
    if connection.is_connected():
            # Create a cursor object for executing SQL queries
            cursor = connection.cursor()
            sql_query = "DELETE from crm_chat2 WHERE tread = "+str(thread_id)
            # Execute the SQL query
            cursor.execute(sql_query)
            connection.commit()
            return {"status_code":200,"detail":"valid authorization","message": f"tread {thread_id} deleted"}