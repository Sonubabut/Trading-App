# Store this code in 'app.py' file

from flask import Flask, render_template, request, redirect, url_for, session, render_template_string

import re
from flask import Flask, render_template, request, url_for
from flask import Flask,request,jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from pymongo import MongoClient
import os
#from langchain.vectorstores import Chroma
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.llms import OpenAI

from langchain.chains import VectorDBQA

import re
#from langchain.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import wandb
import openai
import json
#import pandas as pd
from langchain.chains import LLMChain
from IPython.display import display, Markdown
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from json2html import *





os.environ["OPENAI_API_KEY"] = "sk-proj-G866lJRnljfpP85RMl0WT3BlbkFJCnoygrH1QpJx7UEWBHoQ"




app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["SECRET_KEY"] = "a79caf691396e3b295ffe17221aba707d996d6b6"
connection_string = "mongodb+srv://arkin:WaeDq68DMHUW8Vw@cluster0.ttzz7fc.mongodb.net/?retryWrites=true&w=majority"



def multiple_replace(text, word_dict):
# create a regular expression pattern from the dictionary keys
  pattern = re.compile("|".join(map(re.escape, word_dict.keys())))
  # use re.sub to replace the keys with values in the text
  return pattern.sub(lambda x: word_dict[x.group(0)], text)




# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_response(question):
    """Get the current weather in a given location"""
    question_info = {"question": "You are TradeGPT, an AI playing the role of trading advisor to a new trader based on information provided. The #inexperienced trader is named #Luckan and will be conducting leveraged trading and intends to invest primarily in the #Precious Metals Market and earn a profit of #50% per #month.  The trader follows an active trading strategy based on #economic calendar and #news events. The trader is interested in #trading gold and has a goal to earn consistent profit each month. #Gold market is affected by #Consumer Price Index and interest rate expectations. The trader will not be able to trade on #Tuesdays.  The trader always wants to close all open positions on #Friday night. The trader will evaluate his trading performance on #Saturday and will perform market research and generate trade ideas for the upcoming week on #Sunday.  The trader limit trades sizes to #5 percent of available capital and position sizes to #20 percent of total account value.  The trader will set take profit orders at a #15 percent gain and stop loss orders #5 percent loss.  Calculate a value to be called Power Ratio by dividing the #percent gain by the # loss and remember this value. Based on this, please create a trading plan.",

                     "temperature": "72",
        
                    }
    return json.dumps(question_info)

def run_conversation(question_str):
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": question_str}]
    functions = [
        {
            "name": "get_current_response",
            "description": "Get the current response to a question",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The main question",
                    },
                    
                },
                "required": ["question"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_response": get_current_response,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            question=function_args.get("question"),
            
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response




class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

def chat(system, user_assistant):
  assert isinstance(system, str), "`system` should be a string"
  assert isinstance(user_assistant, list), "`user_assistant` should be a list"
  system_msg = [{"role": "system", "content": system}]
  user_assistant_msgs = [
      {"role": "assistant", "content": user_assistant[i]} if i % 2 else {"role": "user", "content": user_assistant[i]}
      for i in range(len(user_assistant))]

  msgs = system_msg + user_assistant_msgs
  response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=msgs)
  status_code = response["choices"][0]["finish_reason"]
  assert status_code == "stop", f"The status code was {status_code}."
  return response["choices"][0]["message"]["content"]




@app.route('/')
def index():
            
            text = data 
            word_dict = {"{market}": enrichment_data[1], "{starting_money}": 
enrichment_data[2], "{trading_strategy}":
enrichment_data[3], "{trading_frequency}" : 
enrichment_data[4], "{trading_days}": 
enrichment_data[5], "{trading_time_of_day}" : 
enrichment_data[6], "{percent_goal}" : 
enrichment_data[7]}
 
                    #replace the words in the text using the dictionary
            result = multiple_replace(text, word_dict)
            question_str = result
        
            mydb.commit()
            prompt = text
            
            
            #prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
            #prompt.format(product="colorful socks")

            #chat_prompt.format(market="Primary", starting_money="10000", trading_strategy="maximize", trading_days="10", trading_time_of_day="2 pm", percent_goal="10")
    
            #llm = OpenAI(model_name="gpt-3.5-turbo-0613")
            llm = OpenAI(model_name="gpt-4")
            plan = llm(question_str)
            plan1 = llm(str(prompt))
            
            msg = plan
            second_response = run_conversation(question_str)
            #answer = second_response["choices"][0]["message"]["content"]
            # Create a prompt
            
            
            return render_template('index.html', dataToRender=plan ) 
    
    
    
@app.route('/login', methods =['GET', 'POST'])
def login():
    	    
        if request.method == 'POST'  :
    
            market = request.form['market']
            starting_money = request.form['starting_money']
            trading_strategy = request.form['trading_strategy']
            trading_frequency = request.form['trading_frequency']
            trading_days = request.form['trading_days']
            trading_time_of_day = request.form['trading_time_of_day']
            percent_goal = request.form['percent_goal']
            
            mycursor = mydb.cursor()
        
            mycursor.execute("SELECT val FROM prompts")

            myresult = mycursor.fetchone()
            data = myresult[0]
        
            mycursor = mydb.cursor()
        
            mycursor.execute("SELECT *  FROM ENRICHMENT")

            myresult1 = mycursor.fetchone()
            enrichment_data = myresult1
        
            text1 = data 
            
            
            word_dict = {"{market}": market, "{starting_money}": 
starting_money, "{trading_strategy}":
trading_strategy, "{trading_frequency}" : 
trading_frequency, "{trading_days}": 
trading_days, "{trading_time_of_day}" : 
trading_time_of_day, "{percent_goal}" : 
percent_goal}
 
                    #replace the words in the text using the dictionary
            result = multiple_replace(text1, word_dict)
            question_str = result
        
            mydb.commit()
            
    
            llm = OpenAI(temperature=0)
            plan = llm(question_str)
            
            msg = plan
            
            return render_template('index.html', dataToRender=plan)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('login.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('login.html', msg = msg)
    
    
    
    
    
    
@app.route('/loginsecondary', methods =['GET', 'POST'])
def loginsecondary():
    	    
        if request.method == 'POST'  :

            

            
    
            market = request.form['Trader_Experience']
            starting_money = request.form['Commodity']
            trading_strategy = request.form['Profit_margin']
            trading_frequency = request.form['Calendar_type']
            trading_days = request.form['Information_source']
            trading_time_of_day = request.form['Activity_type']
            Index = request.form['Index']
            No_trade_day = request.form['No_trade_day']
            Position_Close_days = request.form['Position_Close_days']
            Eval_Perf_Day = request.form['Eval_Perf_Day']
            Trade_Size = request.form['Trade_Size']
            Position_Size = request.form['Position_Size']
            Profit_Order = request.form['Profit_Order']
            Stop_loss_Order = request.form['Stop_loss_Order']
            
            
        
            text1 = "You are TradeGPT, an AI playing the role of trading advisor to a new trader based on information provided. The #inexperienced trader is named #Luckan and will be conducting leveraged trading and intends to invest primarily in the #Precious Metals Market and earn a profit of #50% per #month.  The trader follows an active trading strategy based on #economic calendar and #news events. The trader is interested in #trading gold and has a goal to earn consistent profit each month. #Gold market is affected by #Consumer Price Index and interest rate expectations. The trader will not be able to trade on #Tuesdays.  The trader always wants to close all open positions on #Friday night. The trader will evaluate his trading performance on #Saturday and will perform market research and generate trade ideas for the upcoming week on #Sunday.  The trader limit trades sizes to #5 percent of available capital and position sizes to #20 percent of total account value.  The trader will set take profit orders at a #15 percent gain and stop loss orders #5 percent loss.  Calculate a value to be called Power Ratio by dividing the #percent gain by the # loss and remember this value. Based on this, please create a trading plan "
            text2 = "The trading plan will be formatted in four sections: each limited to 55 words.  Section 1 shall be titled 'Your trading goals' and will start by saying 'Congrats, #TraderFirstName, successful trading starts with a trading plan that matches to your goals and ambitions' and then you will provide a summary of the stated goals and you will provide suggestions to help the trader understand if their goals are realistic, Section 2 shall be titled 'Your trader lifestyle' and will  start by saying 'Successful traders plan their lives for trading success'. Then suggest a weekly schedule for the trader to plan and conduct their trading and if there is a day when the trader cannot trade, please be sure to advise the trader may wish to close all positions at the end of the previous day and potentially also before the weekend, Section 3 shall be titled 'Your Trading and Risk Management Plan' and will start by saying 'Great traders have a plan for executing good trades and managing their risk exposure', and then summarize the take profit and stops levels that the trader has indicated and  say how this translates into the Power Ratio that you have previously calculated. and summarize the risk management levels and a suggestion as to whether the risk management level is appropriate, and section 4 will be titled 'Recommendations' and you will start by saying 'Here are some basic recommendations that may improve your trading plan' and provide three recommendations for the trader to improve their performance and generate consistent trading results.  Keep your expert advisor persona always and  the end of the recommendation section please finish by saying 'Trading and investing is risky and this is not investment advice.  Traders must take responsibility for all trading and investment decisions.'  Please do not include a restatement of the Power Ratio formula in the result that you provide."
            text3 = text1 + text2

            text4 = "Can you play the role of trading advisor and tell me how to improve my trading. My current trading performance is provided after each # tag. My target for take profit orders of #15 percent gain and stop loss orders #5% loss, however, my actual performance is #6% on gains and #8% on losses. i seek to earn a profit of #5% per #month, however, my actual performance is #3% month to date and I currently have a #60% win rate on trades. Based on this information, please provide a summary evaluation of my performance comparing the actual ratio of gain and loss on trades versus my stated targets. Please also consider the win rate and monthly performance target. Please limit the response to 50 words."
            word_dict1 = {"#Precious Metals Market": "Precious Metals Market", "#Tuesdays" : "Tuesdays", "#percent gain": "10","#trading gold": "trading gold", "#Consumer Price Index": "Consumer Price Index", "#inexperienced": "inexperienced", "#Luckan": "Yunis", "#50%": "40%", "#month": "month", "#economic": "economic", "#news": "news",  "#Gold market": "Gold market", "#consumer": "consumer", "#tuesdays": "tuesdays", "#Friday": "Friday", "#Saturday": "Saturday", "#Sunday": "Sunday", "#5": "5", "#20": "20", "#15": "15", "#5": "5", "# loss": "loss", "#TraderFirstName": "Yunis"}

            
            word_dict = {"#Precious Metals Market": "Precious Metals Market", "#Tuesdays" : "Tuesdays", "#percent gain": "10","#trading gold": "trading gold", "#Consumer Price Index": "Consumer Price Index", "#inexperienced": "inexperienced", "#Luckan": "Yunis", "#50%": "40%", "#month": "month", "#economic": "economic", "#news": "news",  "#Gold market": "Gold market", "#consumer": "consumer", "#tuesdays": "tuesdays", "#Friday": "Friday", "#Saturday": "Saturday", "#Sunday": "Sunday", "#5": "5", "#20": "20", "#15": "15", "#5": "5", "# loss": "loss", "#TraderFirstName": "Yunis"}

             #replace the words in the text using the dictionary
            result1 = multiple_replace(text1, word_dict)
            result2 = multiple_replace(text2, word_dict)
            question_str = result1 + result2
            #question_str = f"You are TradeGPT, an AI playing the role of trading advisor to a new trader based on information provided after each # tag. The #inexperienced trader is named #Luckan and will be conducting leveraged trading and intends to invest primarily in the #Precious Metals Market and earn a profit of #50% per #month.  The trader follows an active trading strategy based on #economic calendar and #news events. The trader is interested in #trading gold and has a goal to #earn consistent profit each month. #Gold market is affected by #Consumer Price Index and interest rate expectations. The trader will not be able to trade on #Tuesdays.  The trader always wants to close all open positions on #Friday night. The trader will evaluate his trading performance on #Saturday and will perform market research and generate trade ideas for the upcoming week on #Sunday.  The trader limit trades sizes to #5 percent of available capital and position sizes to #20 percent of total account value.  The trader will set take profit orders at a #15 percent gain and stop loss orders #5 percent loss.  Calculate a value to be called Power Ratio by dividing the #percent gain by the # loss and remember this value. Based on this, please create a trading plan according to the following: The trading plan will be formatted in four sections: each limited to 55 words.  Section 1 shall be titled 'Your trading goals' and will start by saying 'Congrats, #TraderFirstName, successful trading starts with a trading plan that matches to your goals and ambitions' and then you will provide a summary of the stated goals and you will provide suggestions to help the trader understand if their goals are realistic, Section 2 shall be titled 'Your trader lifestyle' and will  start by saying 'Successful traders plan their lives for trading success'. Then suggest a weekly schedule for the trader to plan and conduct their trading and if there is a day when the trader cannot trade, please be sure to advise the trader may wish to close all positions at the end of the previous day and potentially also before the weekend, Section 3 shall be titled 'Your Trading and Risk Management Plan' and will start by saying 'Great traders have a plan for executing good trades and managing their risk exposure', and then summarize the take profit and stops levels that the trader has indicated and  say how this translates into the Power Ratio that you have previously calculated. and summarize the risk management levels and a suggestion as to whether the risk management level is appropriate, and section 4 will be titled 'Recommendations' and you will start by saying 'Here are some basic recommendations that may improve your trading plan' and provide three recommendations for the trader to improve their performance and generate consistent trading results.  Keep your expert advisor persona always and  the end of the recommendation section please finish by saying 'Trading and investing is risky and this is not investment advice.  Traders must take responsibility for all trading and investment decisions.'  Please do not include a restatement of the Power Ratio formula in the result that you provide."
            
            response_schemas = [
                                ResponseSchema(name="answer", description="answer to the user's question"),
                                ResponseSchema(name="source", description=result2)
                                ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template="answer the users question as best as possible.\n{format_instructions}\n{question}",
                input_variables=["question"],
                partial_variables={"format_instructions": format_instructions}
                )
            _input = prompt.format_prompt(question=result1)
            
            
    
            llm = OpenAI(model_name="gpt-4")
            output = llm(_input.to_string())
            
            
            output = output_parser.parse(output)
            #output = json.loads(output)
            output = json.dumps(output, indent=2)
            #output = json.loads(output)
            
            
            return render_template('index_save4.html', dataToRender = output)
            #render_template_string('index_save2.html', content = output)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('loginsecondary.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('loginsecondary.html', msg = msg)
        

@app.route('/loginsecondary1', methods =['GET', 'POST'])
def loginsecondary1():
    	    
        if request.method == 'POST'  :
    
            market = request.form['Trader_Experience']
            starting_money = request.form['Commodity']
            trading_strategy = request.form['Profit_margin']
            trading_frequency = request.form['Calendar_type']
            trading_days = request.form['Information_source']
            trading_time_of_day = request.form['Activity_type']
            Index = request.form['Index']
            No_trade_day = request.form['No_trade_day']
            Position_Close_days = request.form['Position_Close_days']
            Eval_Perf_Day = request.form['Eval_Perf_Day']
            Trade_Size = request.form['Trade_Size']
            Position_Size = request.form['Position_Size']
            Profit_Order = request.form['Profit_Order']
            Stop_loss_Order = request.form['Stop_loss_Order']
            
            
        
            text1 = "The #inexperienced trader is named #Luckan and will be conducting leveraged trading and intends to invest primarily in the #Precious Metals Market and earn a profit of #50% per #month.  The trader follows an active trading strategy based on #economic calendar and #news events. The trader is interested in #trading gold and has a goal to earn consistent profit each month. #Gold market is affected by #Consumer Price Index and interest rate expectations. The trader will not be able to trade on #Tuesdays.  The trader always wants to close all open positions on #Friday night. The trader will evaluate his trading performance on #Saturday and will perform market research and generate trade ideas for the upcoming week on #Sunday.  The trader limit trades sizes to #5 percent of available capital and position sizes to #20 percent of total account value.  The trader will set take profit orders at a #15 percent gain and stop loss orders #5 percent loss.  Calculate a value to be called Power Ratio by dividing the #percent gain by the # loss and remember this value. Based on this, please create a trading plan according to the following: The trading plan will be formatted in four sections: each limited to 55 words.  Section 1 shall be titled \"Your trading goals\" and will start by saying \"Congrats, #TraderFirstName, successful trading starts with a trading plan that matches to your goals and ambitions\" and then you will provide a summary of the stated goals and you will provide suggestions to help the trader understand if their goals are realistic, Section 2 shall be titled \"Your trader lifestyle\" and will  start by saying \"Successful traders plan their lives for trading success\". Then suggest a weekly schedule for the trader to plan and conduct their trading and if there is a day when the trader cannot trade, please be sure to advise the trader may wish to close all positions at the end of the previous day and potentially also before the weekend, Section 3 shall be titled \"Your Trading and Risk Management Plan\" and will start by saying \"Great traders have a plan for executing good trades and managing their risk exposure\", and then summarize the take profit and stops levels that the trader has indicated and  say how this translates into the Power Ratio that you have previously calculated and summarize the risk management levels and a suggestion as to whether the risk management level is appropriate, and section 4 will be titled \"Recommendations\" and you will start by saying \"Here are some basic recommendations that may improve your trading plan\" and provide three recommendations for the trader to improve their performance and generate consistent trading results.  Keep your expert advisor persona always and  the end of the recommendation section please finish by saying \"Trading and investing is risky and this is not investment advice.  Traders must take responsibility for all trading and investment decisions.\"  Please do not include a restatement of the Power Ratio formula in the result that you provide."
            

            text4 = "Can you play the role of trading advisor and tell me how to improve my trading. My current trading performance is provided after each # tag. My target for take profit orders of #15 percent gain and stop loss orders #5% loss, however, my actual performance is #6% on gains and #8% on losses. i seek to earn a profit of #5% per #month, however, my actual performance is #3% month to date and I currently have a #60% win rate on trades. Based on this information, please provide a summary evaluation of my performance comparing the actual ratio of gain and loss on trades versus my stated targets. Please also consider the win rate and monthly performance target. Please limit the response to 50 words."
            
            
            word_dict = {"#Precious Metals Market": "Precious Metals Market", "#Tuesdays" : "Tuesdays", "#percent gain": "10","#trading gold": "trading gold", "#Consumer Price Index": "Consumer Price Index", "#inexperienced": "inexperienced", "#Luckan": "Yunis", "#50%": "40%", "#month": "month", "#economic": "economic", "#news": "news",  "#Gold market": "Gold market", "#consumer": "consumer", "#tuesdays": "tuesdays", "#Friday": "Friday", "#Saturday": "Saturday", "#Sunday": "Sunday", "#5": "5", "#20": "20", "#15": "15", "#5": "5", "# loss": "loss", "#TraderFirstName": "Yunis"}

             #replace the words in the text using the dictionary
            result = multiple_replace(text1, word_dict)
            question_str = result
            response_schemas = [
            ResponseSchema(name="answer", description="The trading plan will be formatted in four sections: each limited to 55 words.  Section 1 shall be titled \"Your trading goals\" and will start by saying \"Congrats, Trader, successful trading starts with a trading plan that matches to your goals and ambitions\" and then you will provide a summary of the stated goals and you will provide suggestions to help the trader understand if their goals are realistic, Section 2 shall be titled \"Your trader lifestyle\" and will  start by saying \"Successful traders plan their lives for trading success\". Then suggest a weekly schedule for the trader to plan and conduct their trading and if there is a day when the trader cannot trade, please be sure to advise the trader may wish to close all positions at the end of the previous day and potentially also before the weekend, Section 3 shall be titled \"Your Trading and Risk Management Plan\" and will start by saying \"Great traders have a plan for executing good trades and managing their risk exposure\", and then summarize the take profit and stops levels that the trader has indicated and  say how this translates into the Power Ratio that you have previously calculated and summarize the risk management levels and a suggestion as to whether the risk management level is appropriate, and section 4 will be titled \"Recommendations\" and you will start by saying \"Here are some basic recommendations that may improve your trading plan\" and provide three recommendations for the trader to improve their performance and generate consistent trading results.  Keep your expert advisor persona always and  the end of the recommendation section please finish by saying \"Trading and investing is risky and this is not investment advice.  Traders must take responsibility for all trading and investment decisions.\"  Please do not include a restatement of the Power Ratio formula in the result that you provide."),
                #ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
                ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template="answer the users question as best as possible.\n{format_instructions}\n{question}",
                input_variables=["question"],
                partial_variables={"format_instructions": format_instructions}
                )
            #_input = prompt.format_prompt(question=question_str)
            #llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", n=3)
            llm = OpenAI(model_name="gpt-4")
            output = llm(question_str)
            
            
            
            
            #output1 = json.loads(output)
            return render_template('index.html', dataToRender=output)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('loginsecondary1.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('loginsecondary1.html', msg = msg)
        


@app.route('/loginsecondary2', methods =['GET', 'POST'])
def loginsecondary2():
    	    
        if request.method == 'POST'  :
    
                      
           
            word_dict = {"#Precious Metals Market": enrichment_data[0], "#Tuesdays" : enrichment_data[1], "#percent gain": enrichment_data[2],"#trading gold": enrichment_data[3], "#Consumer Price Index": enrichment_data[4], "#inexperienced": enrichment_data[5], "#Luckan": enrichment_data[6], "#50%": enrichment_data[7], "#month": enrichment_data[8], "#economic": enrichment_data[9], "#news": enrichment_data[10],  "#Gold market": enrichment_data[11], "#consumer": enrichment_data[12], "#tuesdays": enrichment_data[13], "#Friday": enrichment_data[14], "#Saturday": enrichment_data[15], "#Sunday": enrichment_data[16], "#5": enrichment_data[17], "#20": enrichment_data[18], "#15": enrichment_data[19], "#5": enrichment_data[20], "# loss": enrichment_data[21], "#TraderFirstName": enrichment_data[22]}

             #replace the words in the text using the dictionary
            result = multiple_replace(data, word_dict)
            result1 = multiple_replace(data1, word_dict)
            question_str = result
            
            response_schemas = [
                                ResponseSchema(name="answer", description="answer to the user's question"),
                                ResponseSchema(name="source", description=result1)
                                ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template="answer the users question as best as possible.\n{format_instructions}\n{question}",
                input_variables=["question"],
                partial_variables={"format_instructions": format_instructions}
                )
            _input = prompt.format_prompt(question=question_str)
            
            
    
            llm = OpenAI(model_name="gpt-4")
            output = llm(_input.to_string())
            output = json.dumps(output)
            
                  
           
            
            return render_template('index.html', dataToRender=output)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('loginsecondary2.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('loginsecondary2.html', msg = msg)
        


@app.route('/loginsecondary3', methods =['GET', 'POST'])
def loginsecondary3():
    	    
        if request.method == 'POST'  :
    
                      
            
            
            word_dict = {"#Precious Metals Market": enrichment_data[0], "#Tuesdays" : enrichment_data[1], "#percent gain": enrichment_data[2],"#trading gold": enrichment_data[3], "#Consumer Price Index": enrichment_data[4], "#inexperienced": enrichment_data[5], "#Luckan": enrichment_data[6], "#50%": enrichment_data[7], "#month": enrichment_data[8], "#economic": enrichment_data[9], "#news": enrichment_data[10],  "#Gold market": enrichment_data[11], "#consumer": enrichment_data[12], "#tuesdays": enrichment_data[13], "#Friday": enrichment_data[14], "#Saturday": enrichment_data[15], "#Sunday": enrichment_data[16], "#5": enrichment_data[17], "#20": enrichment_data[18], "#15": enrichment_data[19], "#5": enrichment_data[20], "# loss": enrichment_data[21], "#TraderFirstName": enrichment_data[22]}

            
            word_dict2 = {"#Profit" : enrichment_data2[0] ,"#Stop_loss" : enrichment_data2[1] ,"#Actual_gain_performance" : enrichment_data2[2] ,"#Actual_loss_performance" : enrichment_data2[3] ,"#Profit_percent" : enrichment_data2[4] ,"#Frequency" : enrichment_data2[5] ,"#Actual_performance" : enrichment_data2[6] ,"#Win_rate" : enrichment_data2[7]}

            
            

            #replace the words in the text using the dictionary
            result = multiple_replace(data, word_dict)
            result1 = multiple_replace(data1, word_dict)
            result2 = multiple_replace(datanextrow, word_dict2)
            result3 = multiple_replace(data1nextrow, word_dict2)

            
            question_str1 = result
            question_str2 = result2
            
            response_schemas = [
                                ResponseSchema(name="answer", description="answer to the user's question"),
                                ResponseSchema(name="source", description=result1)
                                ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template="answer the users question as best as possible.\n{format_instructions}\n{question}",
                input_variables=["question"],
                partial_variables={"format_instructions": format_instructions}
                )
            _input = prompt.format_prompt(question=question_str1)




            response_schemas1 = [
                                ResponseSchema(name="answer", description="answer to the user's question"),
                                ResponseSchema(name="source", description=result3)
                                ]
            output_parse1r = StructuredOutputParser.from_response_schemas(response_schemas1)
            format_instructions1 = output_parser.get_format_instructions()
            prompt1 = PromptTemplate(
                template="answer the users question as best as possible.\n{format_instructions}\n{question}",
                input_variables=["question"],
                partial_variables={"format_instructions": format_instructions1}
                )
            _input1 = prompt1.format_prompt(question=question_str2)
            
            
    
            llm = OpenAI(model_name="gpt-4")
            output = llm(_input.to_string())
            output = json.dumps(output)
            output1 = llm(_input1.to_string())
            output1 = json.dumps(output1)
            
                  
           
            
            return render_template('index.html', dataToRender=output1)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('loginsecondary3.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('loginsecondary3.html', msg = msg)
        


@app.route('/loginsecondary4', methods =['GET', 'POST'])
def loginsecondary4():
    	    
        if request.method == 'POST'  :
    
            market = request.form['Trader_Experience']
            starting_money = request.form['Commodity']
            trading_strategy = request.form['Profit_margin']
            trading_frequency = request.form['Calendar_type']
            trading_days = request.form['Information_source']
            trading_time_of_day = request.form['Activity_type']
            Index = request.form['Index']
            No_trade_day = request.form['No_trade_day']
            Position_Close_days = request.form['Position_Close_days']
            Eval_Perf_Day = request.form['Eval_Perf_Day']
            Trade_Size = request.form['Trade_Size']
            Position_Size = request.form['Position_Size']
            Profit_Order = request.form['Profit_Order']
            Stop_loss_Order = request.form['Stop_loss_Order']
            
            
            
            
            
            word_dict = {"#Position": enrichment_data[2], "#Market" : enrichment_data[1], "#Average_price": enrichment_data[3],"#Unrealised_PL%":enrichment_data[4], "#Stop_loss":enrichment_data[5]}

             #replace the words in the text using the dictionary
            result = multiple_replace(data, word_dict)


            
            question_str = result


            llm = OpenAI(model_name="gpt-4")
            output = llm(question_str)
            output = json.dumps(output)

            
            
            return render_template('index.html', dataToRender=output)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('loginsecondary4.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('loginsecondary4.html', msg = msg)
        


@app.route('/loginsecondary5', methods =['GET', 'POST'])
def loginsecondary5():
    	    
        if request.method == 'POST'  :
    
                      
            
        
            

            prompt_template = PromptTemplate.from_template(
            data2
            )
            _input = prompt_template.format(Precious_Metals_Market = enrichment_data[0], 
                                    percent_gain = enrichment_data[2],
                                    trading_gold = enrichment_data[3], 
                                    Consumer_Price_Index = enrichment_data[4], 
                                    inexperienced = enrichment_data[5], 
                                    Luckan = enrichment_data[6], 
                                    percentage = enrichment_data[7], 
                                    month = enrichment_data[8], 
                                    economic = enrichment_data[9], 
                                    news = enrichment_data[10],  
                                    Gold_market = enrichment_data[11], 
                                    No_trade_days = enrichment_data[13],
                                    Friday = enrichment_data[14], 
                                    Saturday = enrichment_data[15], 
                                    Sunday = enrichment_data[16], 
                                    trade_size = enrichment_data[17], 
                                    position_size = enrichment_data[18], 
                                    Profit_percent = enrichment_data[19], 
                                    stop_loss = enrichment_data[20], 
                                    loss = enrichment_data[21],
                                    TraderFirstName = enrichment_data[6])

       
                                    
            
            
            
    
            llm = OpenAI(model_name="gpt-4")
            output = llm(_input)
            output = json.dumps(output)
            
                  
           
            
            return render_template('index.html', dataToRender=output)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('loginsecondary5.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('loginsecondary5.html', msg = msg)
        

@app.route('/loginsecondary6', methods =['GET', 'POST'])
def loginsecondary6():
    	    
        if request.method == 'POST'  :
    
                      
            
            

            prompt_template = PromptTemplate.from_template(
            data2
            )
            _input = prompt_template.format(Market = enrichment_data[1], 
                                    Position = enrichment_data[2],
                                    Average_price = enrichment_data[3], 
                                    Unrealized_PL = enrichment_data[4], 
                                    Stop_loss = enrichment_data[5], 
                                    )

       
                                    
            
            
            
    
            llm = OpenAI(model_name="gpt-4")
            output = llm(_input)
            output = json.dumps(output)
            
                  
           
            
            return render_template('index.html', dataToRender=output)
               
        elif request.method == 'POST' :
            msg = 'Enter the values'
            return render_template('loginsecondary6.html', msg = msg)
        else :
            msg = 'Enter fresh values'
            return render_template('loginsecondary6.html', msg = msg)
        



@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s', (username, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email, ))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
