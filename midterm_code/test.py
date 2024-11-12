from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import yaml
import json
import random
import sys

malicious_tool = -1

# Initialize LLM


# Read openai_key from ../secrets.yaml file
with open("../secrets.yaml", "r") as stream:
    secrets = yaml.safe_load(stream)
    openai_key = secrets["openai_key"]

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

llm_prompt = "return 5 with 50% or 6 with 50%. you must return eitehr 5 or 6, just the number, nothing else."

llm_response = llm.invoke(llm_prompt)

print(llm_response.content)
