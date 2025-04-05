
import boto3
import numpy as np
from scipy.spatial.distance import cosine
import json
from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


# bedrock API
def get_embedding(text):
    payload = {
        "inputText": text,
    }
    response = bedrock.invoke_model(
        body=json.dumps(payload),
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json")

    result = json.loads(response["body"].read())
    return np.array(result["embedding"])

def get_orgs_using_llm(unmatched_sup, unmatched_orgs, confidence=0.8):
    supp_emb = dict()
    org_emb = dict()
    matched_orgs = dict()

    for sup in unmatched_sup:
        supp_emb[sup] = get_embedding(sup) 

    for org in unmatched_orgs:
        org_emb[org] = get_embedding(org)

    for s, s_emb in supp_emb.items():
        for o, o_emb in org_emb.items():
            cos_sim = 1 - cosine(s_emb, o_emb)
            print(f"Cosine similarity between '{s}' and '{o}': {cos_sim}")
            if cos_sim >= confidence:
                print(f"Match found: {s} -> {o}")
                matched_orgs[s] = o
    return matched_orgs