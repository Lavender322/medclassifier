import os
import re 
import time
import json
import logging
import pickle
import string 
from datetime import datetime
from typing import List, Any, Optional
from fastapi import Depends, APIRouter, FastAPI
from pydantic import BaseModel
import numpy as np
import transformers
import torch
from config import *
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AutoModel, AutoTokenizer, AutoConfig

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
app = FastAPI()

################# Request and response schema #################
class Request(BaseModel):
    query: str
class Response(BaseModel):
    labels: List[str] # Is this okay?
##############################################################
class BERTClass7(torch.nn.Module):
    def __init__(self):
        super(BERTClass7, self).__init__()
        model = AutoModel.from_pretrained(MODEL_FOLDER)
        self.l1 = model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 7)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class CustomDataset(Dataset):

    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation = True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

class Classifier:
    def __init__(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FOLDER, config=AutoConfig.from_pretrained(MODEL_FOLDER))
        self.eval_params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 0
                }
        self.model1 = BERTClass7()
        self.model1.load_state_dict(torch.load(WEIGHTS_FILE, map_location=self.device))
        self.model1.eval()
        self.model1 = self.model1.to(self.device)
        self.model2 = BERTClass7()
        self.model2.load_state_dict(torch.load(WEIGHTS_FILE_NO_SUMMARY, map_location=self.device))
        self.model2.eval()
        self.model2 = self.model2.to(self.device)
        self.MAX_LEN = 512

    def convert_outputs(self, outputs, no_summary=False):
        outputs = np.squeeze(outputs).tolist()
        categories = []
        if outputs[0] == True:
            if no_summary == False:
                categories.append("Overview")
            else:
                categories.append("Epidemiology")
        if outputs[1] == True:
            categories.append("Presentation")
        if outputs[2] == True:
            categories.append("Diagnosis")
        if outputs[3] == True:
            categories.append("Management")
        if outputs[4] == True:
            categories.append("Medications")
        if outputs[5] == True:
            categories.append("Follow up")
        if outputs[6] == True:
            categories.append("Others")
            
        return categories

    def evaluate(self, text, no_summary=False):

        if isinstance(text, str):
            eval_set = CustomDataset(text, self.tokenizer, self.MAX_LEN)
            eval_loader = DataLoader(eval_set, **self.eval_params)

            for new_data in eval_loader:

                ids = new_data['ids'].to(self.device, dtype = torch.long)
                mask = new_data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = new_data['token_type_ids'].to(self.device, dtype = torch.long)

                if no_summary == False:
                    outputs = self.model1(ids, mask, token_type_ids)
                else:
                    outputs = self.model2(ids, mask, token_type_ids)
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = np.array(outputs) >= 0.5

                return self.convert_outputs(outputs, no_summary=no_summary)
        else:    
            return ["Input text was not a string"]

classifier = Classifier()
def get_classifier():
    return classifier
    
@app.post("/classify", response_model=Response)
def classify(request: Request, classifier: Classifier = Depends(get_classifier)):
    logger.info(f"[{datetime.now()}] Request: {request}")
    start_time = time.time()
    results = classifier.evaluate(request.query)
    time_taken = time.time() - start_time
    logger.info(f"{request.query}: {results}: {time_taken}")
    return Response(labels = results)

@app.post("/classify_nosummary", response_model=Response)
def classify(request: Request, classifier: Classifier = Depends(get_classifier)):
    logger.info(f"[{datetime.now()}] Request: {request}")
    start_time = time.time()
    results = classifier.evaluate(request.query, no_summary=True)
    time_taken = time.time() - start_time
    logger.info(f"{request.query}: {results}: {time_taken}")
    return Response(labels = results)

# curl --header "Content-Type: application/json" --request POST --data '{"query" : "Diagnose with ECG."}' http://127.0.0.1:8000/classify
