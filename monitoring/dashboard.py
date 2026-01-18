import pandas as pd
from fastapi import FastAPI
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

app = FastAPI()

reference = pd.read_csv("monitoring/reference.csv")
current = pd.read_csv("monitoring/current.csv")

@app.get("/")
def run_dashboard():
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report.as_dict()

