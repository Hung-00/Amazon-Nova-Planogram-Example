# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3
import json
from datetime import datetime
import base64
from pathlib import Path
from botocore.exceptions import ClientError


# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")


MODEL_ID = "amazon.nova-pro-v1:0"

# Define your system prompt(s).
system_list = [{"text": "You a Planogram Specialist."}]

image1 = Path("good1.jpg")

# with open(image1, "rb") as image_file:
#     encoded_image1 = base64.b64encode(image_file.read())

# image2 = Path("good2.jpg")

# with open(image2, "rb") as image_file:
#     encoded_image2 = base64.b64encode(image_file.read())

# image3 = Path("bad1.jpg")

# with open(image3, "rb") as image_file:
#     encoded_image3 = base64.b64encode(image_file.read())

# --------------------------------------------------------------
image = Path("f.jpg")

with open(image, "rb") as image_file:
    evaluate_image = base64.b64encode(image_file.read())


# Define one or more messages using the "user" and "assistant" roles.
message_list = [
    {
        "role": "user",
        "content": [
            # {
            #     "text": "These are Uniben's beverage product:\
            # 1. Yellow bottle with yellow cap is Boncha.\
            # 2. Blue, orange or red bottle with black cap is Abben.\
            # 3. White or purple bottle with white cap is Joco Milk.\
            # 4. Red bottle with red cap is Joco Fruit.\
            # Refrigerator with good planogram follow these 4 conditions:\
            # 1. Each shelf must have 6 or more bottles.\
            # 2. Bottles on each shelf must be in the same product, but not necessarily in the same color.\
            # 3. If the refrigetor have 4 shelves, the order from top to bottom should be  Boncha on the first and second shelves, then Abben on the third shelf, then Joco Milk or Joco Fruit on the bottom shelf. Sometimes Joco Milk and Joco Fruit can be on the same bottom shelf.\
            # Then you need to evaluate the refrigerator PASS or FAILED:\
            # - If all conditions are qualified, then the refrigerator is PASS.\
            # - If any conditions is not qualified, then the refrigerator is FAILED.\
            # Tell me the reason why the refrigerator PASS or FAILED in 'reason' attribute in the JSON format."
            # },
            {
                "image": {
                    "format": "jpeg",
                    # "format": "png",
                    "source": {"bytes": evaluate_image.decode()},
                }
            },
            {
                "text": """You will receive a picture. Answer these questions:
             1. How many shelves?
             2. Describe the products on each shelf. For example: Shelf 1 has one Chivas Regal."""
            },
        ],
    }
]


# Configure the inference parameters.
inf_params = {"maxTokens": 500, "topP": 1, "temperature": 0}

request_body = {
    "schemaVersion": "messages-v1",
    "messages": message_list,
    "system": system_list,
    "inferenceConfig": inf_params,
}

start_time = datetime.now()

# Invoke the model with the response stream
response = client.invoke_model_with_response_stream(
    modelId=MODEL_ID, body=json.dumps(request_body)
)

request_id = response.get("ResponseMetadata").get("RequestId")
print(f"Request ID: {request_id}")
print("Awaiting first token...")

chunk_count = 0
time_to_first_token = None


# Process the response stream
stream = response.get("body")
if stream:
    for event in stream:
        chunk = event.get("chunk")
        if chunk:
            # Print the response chunk
            chunk_json = json.loads(chunk.get("bytes").decode())
            # Pretty print JSON
            # print(json.dumps(chunk_json, indent=2, ensure_ascii=False))
            content_block_delta = chunk_json.get("contentBlockDelta")
            if content_block_delta:
                if time_to_first_token is None:
                    time_to_first_token = datetime.now() - start_time
                    print(f"Time to first token: {time_to_first_token}")

                chunk_count += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                # print(f"{current_time} - ", end="")
                print(content_block_delta.get("delta").get("text"), end="")
    print(f"Total chunks: {chunk_count}")
else:
    print("No response stream received.")
