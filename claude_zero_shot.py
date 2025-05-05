import boto3
import json
import base64
from pathlib import Path
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

image1 = Path("ensure.jpg")

with open(image1, "rb") as image_file:
    evaluate_image_1 = base64.b64encode(image_file.read())

image2 = Path("chips.jpg")

with open(image2, "rb") as image_file:
    evaluate_image_2 = base64.b64encode(image_file.read())

system = [
    {
        "type": "text",
        "text": """
        <instruction>
        You are a Planogram Specialist tasked with analyzing images of product shelves. Your goal is to identify the products displayed, count their quantities, and provide a detailed response to the user.

        Follow these steps:
        1. Carefully examine the provided image of the product shelf.
        2. Identify each distinct product shown on the shelf.
        3. For each identified product, count the total quantity present on the shelf.
        4. Compile your findings into a structured response in the following format:

        <product_list>
        [Product 1 Name]: [Quantity]
        [Product 2 Name]: [Quantity]
        ...
        </product_list>

        Ensure your response is accurate, concise, and follows the specified format exactly. Do not include any additional explanations or assumptions beyond the requested product names and quantities.
        </instruction>
        """,
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": evaluate_image_1.decode("utf-8"),
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": evaluate_image_2.decode("utf-8"),
                },
            },
            {
                "type": "text",
                "text": """
                Analyze the pictures. Seperate the answer for each picture.
                """,
            },
        ],
    }
]

inferenceConfig = {"max_tokens": 500, "top_p": 1, "temperature": 0}

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "system": system,
    "messages": messages,
    "max_tokens": 500,
    "top_p": 1,
    "temperature": 0,
}

try:
    response = client.invoke_model(modelId=modelId, body=json.dumps(body))

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{modelId}'. Reason: {e}")
    exit(1)

model_response = json.loads(response["body"].read())
print(json.dumps(model_response, indent=4))

print(model_response["content"][0]["text"])
