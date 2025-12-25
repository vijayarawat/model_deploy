# import boto3, json

# client = boto3.client("sagemaker-runtime", region_name="us-east-1")

# payload = {
#     "inputs": "Give top 5 automotive lead scoring features",
#     "parameters": {
#         "max_new_tokens": 200,
#         "temperature": 0.6,
#         "top_p": 0.9,
#         "do_sample": True
#     }
# }

# response = client.invoke_endpoint(
#     EndpointName="endpoint-quick-start-pqweq",
#     ContentType="application/json",
#     Body=json.dumps(payload)
# )

# result = json.loads(response["Body"].read().decode())
# print(result)



import boto3
import json
import time

# === CONFIGURE THESE ===
REGION = "us-east-1"

MODEL_SOURCE_ARN = "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/huggingface-textgeneration-bloom-1b1/2.3.12"
ENDPOINT_NAME = "endpoint-quick-start-pqweq"
EXECUTION_ROLE = "arn:aws:iam::163434000843:role/AmazonSageMakerFullAccess"
INSTANCE_TYPE = "ml.g5.xlarge"
INSTANCE_COUNT = 1


# Create Bedrock client
bedrock = boto3.client("bedrock", region_name=REGION)

def deploy():
    print("Deploying model to SageMaker endpoint via Bedrock…")

    config = {
        "sageMaker": {
            "initialInstanceCount": INSTANCE_COUNT,
            "instanceType": INSTANCE_TYPE
        }
    }

    response = bedrock.create_marketplace_model_endpoint(
        modelSourceIdentifier=MODEL_SOURCE_ARN,
        endpointConfig=json.dumps(config),
        endpointName=ENDPOINT_NAME
    )
    
    endpoint_arn = response["marketplaceModelEndpoint"]["endpointArn"]
    print("Endpoint ARN:", endpoint_arn)

    # Wait for status
    while True:
        status_resp = bedrock.get_marketplace_model_endpoint(endpointArn=endpoint_arn)
        status = status_resp["endpointStatus"]
        print("Status:", status)
        if status == "InService":
            print("Deployment complete!")
            break
        print("Waiting for endpoint to be InService…")
        time.sleep(30)

if __name__ == "__main__":
    deploy()
