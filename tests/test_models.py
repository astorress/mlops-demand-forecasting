import mlflow
import pandas as pd

from src.models.evaluate import compare_with_champion, evaluate_model
from src.models.train import train_model


def test_train_and_evaluate(processed_dir, local_mlflow):
    run_id = train_model(
        params={"n_estimators": 50},
        data_path=str(processed_dir),
        experiment_name="test_experiment",
    )
    run = mlflow.get_run(run_id)
    assert run.data.metrics["val_r2"] > 0.8

    metrics = evaluate_model(run_id, test_path=str(processed_dir / "test.parquet"))
    # MAPE en kWh reales, no en unidades normalizadas
    assert metrics["mape"] < 15

    # Sin modelo en Production, el nuevo modelo se promueve
    assert compare_with_champion(run_id, "modelo_inexistente", str(processed_dir / "test.parquet"))


def test_logged_model_predicts_from_raw_features(processed_dir, local_mlflow):
    run_id = train_model(params={"n_estimators": 20}, data_path=str(processed_dir), experiment_name="test_experiment")
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    preds = model.predict(test_df.head(24))
    # Las predicciones están en la escala de la demanda real
    assert abs(preds.mean() - test_df["demand_kwh"].head(24).mean()) < test_df["demand_kwh"].std()
