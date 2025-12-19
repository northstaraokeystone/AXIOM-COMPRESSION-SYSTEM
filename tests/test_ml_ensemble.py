"""Tests for ML ensemble 60s dust prediction.

Tests:
- Config loading
- Ensemble initialization
- Model training
- Ensemble prediction
- Agreement computation
- Accuracy assessment
"""

import pytest
from src.cfd_dust_dynamics import (
    load_ml_ensemble_config,
    initialize_ensemble,
    train_ensemble,
    ensemble_predict,
    compute_ensemble_agreement,
    weighted_ensemble_average,
    ml_ensemble_forecast,
    compute_forecast_accuracy,
    ML_ENSEMBLE_MODEL_COUNT,
    ML_ENSEMBLE_PREDICTION_HORIZON_S,
    ML_ENSEMBLE_ACCURACY_TARGET,
)


class TestMLEnsembleConfig:
    """Tests for ML ensemble configuration."""

    def test_config_loads(self):
        """Config loads successfully."""
        config = load_ml_ensemble_config()
        assert config is not None
        assert "model_count" in config

    def test_model_count(self):
        """Model count is 5."""
        config = load_ml_ensemble_config()
        assert config["model_count"] == 5
        assert ML_ENSEMBLE_MODEL_COUNT == 5

    def test_model_types(self):
        """Model types are correct."""
        config = load_ml_ensemble_config()
        types = config["model_types"]
        assert "lstm" in types
        assert "transformer" in types
        assert "cnn" in types
        assert "xgboost" in types
        assert "gru" in types

    def test_prediction_horizon(self):
        """Prediction horizon is 60s."""
        config = load_ml_ensemble_config()
        assert config["prediction_horizon_s"] == 60
        assert ML_ENSEMBLE_PREDICTION_HORIZON_S == 60

    def test_accuracy_target(self):
        """Accuracy target is 0.90."""
        config = load_ml_ensemble_config()
        assert config["accuracy_target"] == 0.90
        assert ML_ENSEMBLE_ACCURACY_TARGET == 0.90

    def test_agreement_threshold(self):
        """Agreement threshold present."""
        config = load_ml_ensemble_config()
        assert "agreement_threshold" in config
        assert config["agreement_threshold"] == 0.80


class TestEnsembleInitialization:
    """Tests for ensemble initialization."""

    def test_ensemble_initializes(self):
        """Ensemble initializes successfully."""
        ensemble = initialize_ensemble()
        assert ensemble is not None
        assert "models" in ensemble

    def test_model_count_correct(self):
        """Correct number of models."""
        ensemble = initialize_ensemble()
        assert len(ensemble["models"]) == ML_ENSEMBLE_MODEL_COUNT

    def test_all_model_types(self):
        """All model types present."""
        ensemble = initialize_ensemble()
        model_types = [m["type"] for m in ensemble["models"]]

        expected = ["lstm", "transformer", "cnn", "xgboost", "gru"]
        for exp in expected:
            assert exp in model_types

    def test_model_properties(self):
        """Models have required properties."""
        ensemble = initialize_ensemble()
        for model in ensemble["models"]:
            assert "type" in model
            assert "initialized" in model
            assert model["initialized"] is True


class TestEnsembleTraining:
    """Tests for ensemble training."""

    def test_training_runs(self):
        """Training runs successfully."""
        result = train_ensemble()
        assert result is not None
        assert "models_trained" in result

    def test_all_models_trained(self):
        """All models trained."""
        result = train_ensemble()
        assert result["models_trained"] == ML_ENSEMBLE_MODEL_COUNT
        assert result["all_trained"] is True

    def test_training_metrics(self):
        """Training reports metrics."""
        result = train_ensemble()
        assert "model_metrics" in result
        metrics = result["model_metrics"]

        for model_type, m in metrics.items():
            assert "loss" in m
            assert "accuracy" in m

    def test_training_time(self):
        """Training time reported."""
        result = train_ensemble()
        assert "training_time_s" in result
        assert result["training_time_s"] > 0


class TestEnsemblePrediction:
    """Tests for ensemble prediction."""

    def test_prediction_runs(self):
        """Prediction runs."""
        predictions = ensemble_predict(horizon_s=60)
        assert predictions is not None

    def test_prediction_per_model(self):
        """Each model produces prediction."""
        predictions = ensemble_predict(horizon_s=60)
        assert len(predictions) == ML_ENSEMBLE_MODEL_COUNT

    def test_prediction_values(self):
        """Predictions are valid values."""
        predictions = ensemble_predict(horizon_s=60)
        for model, pred in predictions.items():
            assert isinstance(pred, float)


class TestEnsembleAgreement:
    """Tests for ensemble agreement."""

    def test_agreement_computes(self):
        """Agreement computes."""
        predictions = {"a": 0.5, "b": 0.52, "c": 0.48}
        agreement = compute_ensemble_agreement(predictions)

        assert isinstance(agreement, float)
        assert 0 <= agreement <= 1

    def test_high_agreement(self):
        """Similar predictions = high agreement."""
        predictions = {"a": 0.5, "b": 0.5, "c": 0.5}
        agreement = compute_ensemble_agreement(predictions)

        assert agreement > 0.9

    def test_low_agreement(self):
        """Divergent predictions = low agreement."""
        predictions = {"a": 0.0, "b": 0.5, "c": 1.0}
        agreement = compute_ensemble_agreement(predictions)

        assert agreement < 0.7


class TestWeightedAverage:
    """Tests for weighted ensemble average."""

    def test_average_computes(self):
        """Weighted average computes."""
        predictions = {"a": 0.5, "b": 0.6, "c": 0.7}
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        average = weighted_ensemble_average(predictions, weights)

        assert isinstance(average, float)

    def test_equal_weights(self):
        """Equal weights = simple average."""
        predictions = {"a": 0.3, "b": 0.6, "c": 0.9}
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        average = weighted_ensemble_average(predictions, weights)

        expected = (0.3 + 0.6 + 0.9) / 3
        assert abs(average - expected) < 0.001

    def test_weighted(self):
        """Different weights work."""
        predictions = {"a": 0.0, "b": 1.0}
        weights = {"a": 0.25, "b": 0.75}
        average = weighted_ensemble_average(predictions, weights)

        expected = 0.75  # Weighted toward b
        assert abs(average - expected) < 0.001


class TestMLEnsembleForecast:
    """Tests for full ensemble forecast."""

    def test_forecast_runs(self):
        """Forecast runs."""
        result = ml_ensemble_forecast(horizon_s=60)
        assert result is not None

    def test_forecast_properties(self):
        """Forecast has required properties."""
        result = ml_ensemble_forecast(horizon_s=60)

        assert "horizon_s" in result
        assert result["horizon_s"] == 60
        assert "model_count" in result
        assert "predictions" in result
        assert "weighted_prediction" in result

    def test_forecast_agreement(self):
        """Forecast includes agreement."""
        result = ml_ensemble_forecast(horizon_s=60)

        assert "agreement" in result
        assert 0 <= result["agreement"] <= 1

    def test_forecast_accuracy(self):
        """Forecast includes accuracy."""
        result = ml_ensemble_forecast(horizon_s=60)

        assert "accuracy" in result
        assert "accuracy_met" in result

    def test_forecast_target(self):
        """Forecast evaluates against target."""
        result = ml_ensemble_forecast(horizon_s=60)

        target_met = result["accuracy"] >= ML_ENSEMBLE_ACCURACY_TARGET
        assert result["accuracy_met"] == target_met


class TestForecastAccuracy:
    """Tests for forecast accuracy computation."""

    def test_accuracy_computes(self):
        """Accuracy computes."""
        predicted = 0.5
        actual = 0.52
        accuracy = compute_forecast_accuracy(predicted, actual)

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_perfect_accuracy(self):
        """Perfect prediction = 1.0 accuracy."""
        predicted = 0.5
        actual = 0.5
        accuracy = compute_forecast_accuracy(predicted, actual)

        assert accuracy == 1.0

    def test_accuracy_decreases_with_error(self):
        """More error = lower accuracy."""
        acc1 = compute_forecast_accuracy(0.5, 0.55)
        acc2 = compute_forecast_accuracy(0.5, 0.60)

        assert acc1 > acc2


class TestMLEnsembleIntegration:
    """Integration tests for ML ensemble."""

    def test_end_to_end(self):
        """End-to-end ensemble workflow."""
        # Initialize
        ensemble = initialize_ensemble()
        assert ensemble is not None

        # Train
        train_result = train_ensemble()
        assert train_result["all_trained"] is True

        # Predict
        forecast = ml_ensemble_forecast(horizon_s=60)
        assert forecast is not None
        assert "weighted_prediction" in forecast

    def test_60s_horizon(self):
        """60s prediction horizon works."""
        result = ml_ensemble_forecast(horizon_s=60)

        assert result["horizon_s"] == 60
        # This is 2x the previous 30s horizon
        assert ML_ENSEMBLE_PREDICTION_HORIZON_S == 60

    def test_5_models(self):
        """All 5 models contribute."""
        result = ml_ensemble_forecast(horizon_s=60)

        assert result["model_count"] == 5
        assert len(result["predictions"]) == 5
