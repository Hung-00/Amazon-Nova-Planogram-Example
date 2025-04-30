import boto3
import json
import base64
from pathlib import Path
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


# image1 = Path("good1.jpg")

# with open(image1, "rb") as image_file:
#     encoded_image1 = base64.b64encode(image_file.read())

# image2 = Path("good2.jpg")

# with open(image2, "rb") as image_file:
#     encoded_image2 = base64.b64encode(image_file.read())

# --------------------------------------------------------------
image = Path("5.jpg")

with open(image, "rb") as image_file:
    evaluate_image = base64.b64encode(image_file.read())

system = [
    {
        "type": "text",
        "text": "You are a Planogram Specialist for Uniben, which is a beverage company.",
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "These are Uniben's beverage products: 1. Yellow bottle with yellow cap is Boncha. 2. Blue, orange or red bottle with black cap is Abben. 3. White or purple bottle with white cap is Joco Milk. 4. Red bottle with red cap is Joco Fruit. Refrigerator with good planogram follow these 4 conditions: 1. Each shelf must have 6 or more bottles. 2. Bottles on each shelf must be in the same product, but not necessarily in the same color. 3. If the refrigetor have 4 shelves, the order from top to bottom should be  Boncha on the first and second shelves, then Abben on the third shelf, then Joco Milk or Joco Fruit on the bottom shelf. 4. If the refrigetor have 5 shelves, the order from top to bottom should be Boncha on the first and second shelves, then Joco Milk on the third shelf, then Abben on the fourth, then Joco Fruit on the bottom shelf. Sometimes Joco Milk and Joco Fruit can be on the same bottom shelf. Then you need to evaluate the refrigerator PASS or FAILED: 1. If all conditions are qualified, then the refrigerator is PASS. 2. If any conditions is not qualified, then the refrigerator is FAILED. Tell me the reason why the refrigerator PASS or FAILED in 'reason' attribute in the JSON format.",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": evaluate_image.decode("utf-8"),
                },
            },
            {
                "type": "text",
                "text": 'The refrigerator is the one you need to evaluate. Only give me back the response with the JSON format here, replac: \
                    {"refrigerator":{"inventory":[{"shelf":1,"product": ,"quantity": Just count bottles on the front line},{"shelf":2,"product": ,"quantity": },...],"conclusion":{"result":FAILED or PASSED,"reason": Tell me the reason why}}}',
            },
        ],
    }
]

inferenceConfig = {"max_tokens": 500, "top_p": 1, "temperature": 0}

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "system": system,
    "messages": messages,
    # "inferenceConfig": inferenceConfig,
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

raw_text = model_response["content"][0]["text"]

# Remove the code fence markers (e.g., "```json" at the start and "```" at the end)
if raw_text.startswith("```json"):
    cleaned_text = raw_text[len("```json") :].strip()
else:
    cleaned_text = raw_text.strip()

if cleaned_text.endswith("```"):
    cleaned_text = cleaned_text[: -len("```")].strip()

try:
    parsed_json = json.loads(cleaned_text)
    print("Parsed JSON:")
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError as e:
    print("Error decoding JSON:", e)
