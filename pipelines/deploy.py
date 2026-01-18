from kfp.dsl import component

@component(base_image="python:3.10")
def deploy_model(approved: bool):
    if not approved:
        raise RuntimeError("Model did not meet accuracy threshold")

    print("Deploying model...")
    print("kubectl rollout restart deployment ml-fastapi-deployment")

