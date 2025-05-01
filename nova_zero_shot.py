import boto3
import json
from datetime import datetime
import base64
from pathlib import Path
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "amazon.nova-pro-v1:0"

system_list = [{"text": "You a Planogram Specialist."}]


image = Path("choco2.jpg")

with open(image, "rb") as image_file:
    evaluate_image = base64.b64encode(image_file.read())


message_list = [
    {
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": evaluate_image.decode()},
                }
            },
            {
                "text": """
                You will receive a picture. Answer these questions:
             1. How many shelves?
             2. Describe the products on each shelf. Ignore the price.
             """
            },
        ],
    }
]


# Configure the inference parameters.
inf_params = {"maxTokens": 500, "topP": 1, "topK": 10, "temperature": 0}

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
    print(f"Total chunks: {chunk_count}")
else:
    print("No response stream received.")
