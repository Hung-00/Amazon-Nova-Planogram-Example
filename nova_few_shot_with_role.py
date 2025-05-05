import boto3
import json
from datetime import datetime
import base64
from pathlib import Path
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "amazon.nova-pro-v1:0"


system_list = [
    {
        "text": """
            # Task
            You are a Planogram Specialist responsible for analyzing product displays on shelves. Your task is to identify the products, count their quantities, and provide this information to the user.

            ## Instructions
            1. You will receive one or more images of shelves with products displayed on them.
            2. Carefully examine each image and identify the different products present on the shelves.
            3. For each identified product, count the quantity or number of units displayed on the shelves.
            4. Provide your response in the following format:

            ### Product Identification and Quantity
            - Product 1: [Product Name], Quantity: [Number]
            - Product 2: [Product Name], Quantity: [Number]
            - ...

            5. List all the identified products and their corresponding quantities in this structured format.
            6. Do not include any additional explanations or assumptions in your response.

            Provide your response immediately after these instructions, following the specified format.
           """
    }
]

image1 = Path("ensure.jpg")

with open(image1, "rb") as image_file:
    evaluate_image_1 = base64.b64encode(image_file.read())

image2 = Path("xylitol.jpg")

with open(image2, "rb") as image_file:
    evaluate_image_2 = base64.b64encode(image_file.read())

image3 = Path("chips.jpg")

with open(image3, "rb") as image_file:
    evaluate_image_3 = base64.b64encode(image_file.read())


message_list = [
    {
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": evaluate_image_1.decode()},
                }
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "text": """
                - Product 1: Ensure Gold StrengthPro:, Quantity: 3
                - Product 2: Ensure Original Nutrition Shake (Vanilla flavor):, Quantity: 3
                """,
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": evaluate_image_2.decode()},
                }
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "text": """
                - Product 1: Red Xylitol, Quantity: 2
                - Product 2: Purple Xylitol, Quantity: 6
                - Product 3: Blue Xylitol, Quantity: 4
            """,
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": evaluate_image_3.decode()},
                }
            },
            {
                "text": """
                Analyze the third picture.
                """,
            },
        ],
    },
]


inf_params = {"maxTokens": 500, "topP": 1, "temperature": 0}

request_body = {
    "schemaVersion": "messages-v1",
    "messages": message_list,
    "system": system_list,
    "inferenceConfig": inf_params,
}

start_time = datetime.now()

response = client.invoke_model_with_response_stream(
    modelId=MODEL_ID, body=json.dumps(request_body)
)

request_id = response.get("ResponseMetadata").get("RequestId")
print(f"Request ID: {request_id}")
print("Awaiting first token...")

chunk_count = 0
time_to_first_token = None


stream = response.get("body")
if stream:
    for event in stream:
        chunk = event.get("chunk")
        if chunk:

            chunk_json = json.loads(chunk.get("bytes").decode())

            content_block_delta = chunk_json.get("contentBlockDelta")
            if content_block_delta:
                if time_to_first_token is None:
                    time_to_first_token = datetime.now() - start_time
                    print(f"Time to first token: {time_to_first_token}")

                chunk_count += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")

                print(content_block_delta.get("delta").get("text"), end="")
    print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nTotal chunks: {chunk_count}")
else:
    print("No response stream received.")
