import boto3
import json
import base64
from pathlib import Path
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

modelId = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


image1 = Path("good1.jpg")

with open(image1, "rb") as image_file:
    encoded_image1 = base64.b64encode(image_file.read())

image2 = Path("good2.jpg")

with open(image2, "rb") as image_file:
    encoded_image2 = base64.b64encode(image_file.read())

# --------------------------------------------------------------
image = Path("16.jpg")

with open(image, "rb") as image_file:
    evaluate_image = base64.b64encode(image_file.read())

system = [
    {
        "type": "text",
        "text": """## Task
You are a Planogram Specialist for Uniben, a beverage company. Your task is to evaluate the planograms (product placement) in refrigerators containing Uniben's beverage products.
""",
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """
## Instructions
1. Review the provided information about Uniben's beverage products:
### Uniben's Beverage Products
- Yellow bottle with yellow cap is Boncha, usually can only have a maximum of 8 bottles on one shelf. 
- Blue, orange or red bottle with black cap is Abben, usually can only have a maximum of 9 bottles on one shelf. 
- White or purple bottle with white cap is Joco Milk, usually can only have a maximum of 9 bottles on one shelf. 
- Red or purple bottle with red cap is Joco Fruit, usually can only have a maximum of 8 bottles on one shelf. 

Refrigerator with good planogram follow these 4 conditions: 
- Each shelf must have 6 or more bottles on the front row, ignore bottles behind and the shelf talker in front of the shelf. 
- Bottles on each shelf must be in the same product, but not necessarily in the same color. 
- If the refrigetor have 4 shelves, the order from top to bottom should be  Boncha on the first and second shelves, then Abben on the third shelf, then Joco Milk or Joco Fruit on the bottom shelf. 
- If the refrigetor have 5 shelves, the order from top to bottom should be Boncha on the first and second shelves, then Joco Milk on the third shelf, then Abben on the fourth, then Joco Fruit on the bottom shelf. Sometimes Joco Milk and Joco Fruit can be on the same bottom shelf.

2. You will receive 3 pictures of refrigerators containing Uniben's beverage products.
3. Analyze the first two refrigerator pictures to understand how and where specific beverage products should be placed (good planograms).
4. Evaluate the third refrigerator's planogram based on the following conditions:
    - All Uniben's beverage products are present and correctly placed according to the good planograms.
    - Products are facing forward and properly aligned.
    - Shelves are fully stocked with no empty spaces.
    - Refrigerator is clean and organized.
5. Determine if the third refrigerator's planogram is PASS or FAILED:
    - If all conditions are met, the planogram is PASS.
    - If any condition is not met, the planogram is FAILED.
6. Provide your evaluation in JSON format with a 'reason' attribute explaining why the planogram PASSED or FAILED.
""",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded_image1.decode("utf-8"),
                },
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": '{"refrigerator":{"inventory":[{"shelf":1,"product":"Boncha","quantity":8},{"shelf":2,"product":"Boncha","quantity":8},{"shelf":3,"product":"Joco Fruit","quantity":8},{"shelf":4,"product":"Joco Milk","quantity":9},{"shelf":5,"product":"Abben","quantity":9}],"conclusion":{"result":"FAILED","reason":"Failed because Abben bottles are on the bottom shelf."}}}',
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded_image2.decode("utf-8"),
                },
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": '{"refrigerator":{"inventory":[{"shelf":1,"product":"Boncha","quantity":8},{"shelf":2,"product":"Boncha","quantity":7},{"shelf":3,"product":"Joco Milk","quantity":8},{"shelf":4,"product":"Abben","quantity":8},{"shelf":5,"product":"Joco Fruit","quantity":7}],"conclusion":{"result":"PASS","reason":"Pass all conditions."}}}',
            },
        ],
    },
    {
        "role": "user",
        "content": [
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
                "text": """
                ### Output Format
                The last refrigerator is the one you need to evaluate. Only give me back the JSON object response like above, do not give me anything else.""",
            },
        ],
    },
]


body = {
    "anthropic_version": "bedrock-2023-05-31",
    "system": system,
    "messages": messages,
    "max_tokens": 500,
    "top_p": 0.9,
    "top_k": 20,
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
