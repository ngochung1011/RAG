from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import torch
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM
import logging
import pandas as pd

df = pd.read_csv("/root/source/predict_hscode/before_update/code/india.csv")
df = df[:100]
documents= df['document'].to_list()
description = df['CARGO DESCRIPTION'].to_list()
description = [str(des).lower() for des in description]


llm = ""
prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(
                        """
        Extract the appropriate 6-digit HS Code from the product description by thoroughly analyzing its details and utilizing a reliable and up-to-date HS Code database for accurate results.
        Only return the HS Code as a 6-digit number .
        Example: 123456
        HS Code is a number.
        Context: {context}
        Description: {description}
        Answer:
        """
        )
    ])
    

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    llm = OllamaLLM(model="gemma2", temperature=0, device=device)
except Exception as e:
    logging.error(f"Failed to load model on {device}: {e}")
    llm = OllamaLLM(model="gemma2", temperature=0, device="cpu")  
    

chain = prompt|llm

n = len(df)
result = []


for i in tqdm(range(n)):
    doc = documents[i]
    des = description[i]
    label = chain.run({'context': doc, 'description': des})
    #print(label)
    result.append(label)

df2 = pd.DataFrame()
#df['HS CODES'] = result
df2['HS CODES'] = result
df2['CARGO DESCRIPTION']=description


