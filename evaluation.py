import dill
import pandas as pd
from datetime import datetime

if __name__ == "__main__":

    TIME_LABEL = datetime.now().strftime("_%m%d_%H%M")

    model_file = input("Insert path to the evaluated model:\nmodels/")
    model_path = f'models/{model_file}'
    submission_path = f"output/submission{TIME_LABEL}.csv"

    df = pd.read_csv("data/test.csv")

    with open(model_path, "rb") as file:
        model = dill.load(file)

    y_pred = model.predict_proba(df)[:, 1]

    submission = pd.DataFrame({"id": df['id'], "Exited": y_pred})
    submission.to_csv(submission_path, index=False)
