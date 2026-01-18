from kfp.dsl import pipeline
from train import train_model
from evaluate import evaluate_model
from deploy import deploy_model

@pipeline(name="mlops-pipeline")
def ml_pipeline():
    model_path = "/tmp/model.joblib"

    train = train_model(model_path=model_path)
    eval = evaluate_model(
        model_path=model_path,
        accuracy=0.9
    )

    deploy_model(approved=eval.output)

