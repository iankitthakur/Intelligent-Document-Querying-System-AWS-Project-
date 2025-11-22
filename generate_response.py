import boto3
from botocore.exceptions import ClientError
import json

# Initialize AWS Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

def generate_response(prompt, model_id, temperature, top_p):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500,
                "temperature": temperature,
                "top_p": top_p,
            })
        )
        return json.loads(response['body'].read())['content'][0]["text"]
    except ClientError as e:
        print(f"Error generating response: {e}")
        return ""

response = generate_response("Explain how a crane works.", "anthropic.claude-3-haiku-20240307-v1:0", 0.7, 0.9)
print(response)