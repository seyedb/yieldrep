from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import Tensor

from yieldrep.config import ProjectConfig
from yieldrep.factors.curve import curve_panel


class CurveAutoencoder(nn.Module):
    """Small MLP autoencoder for curve reconstruction, not forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        decoder_output_dim = output_dim if output_dim is not None else input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, decoder_output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(inputs)))


def build_autoencoder(config: ProjectConfig) -> list[Path]:
    """Train per-country autoencoders and write embeddings/reconstructions."""
    curves = pd.read_parquet(config.curves_path)
    output_paths: list[Path] = []
    config.autoencoder_dir.mkdir(parents=True, exist_ok=True)

    for country in sorted(curves["country"].dropna().unique()):
        panel = curve_panel(curves, str(country)).ffill().dropna()
        panel.attrs["country"] = str(country)
        if panel.shape[1] < config.pca.min_maturities:
            continue
        split = _date_ordered_panel_split(panel, config.evaluation.test_fraction)
        if split is None:
            continue
        train_panel, _ = split
        if len(train_panel) < config.autoencoder.min_train_dates:
            continue

        result = fit_autoencoder_panel(
            panel=panel,
            test_fraction=config.evaluation.test_fraction,
            latent_dim=config.autoencoder.latent_dim,
            hidden_dim=config.autoencoder.hidden_dim,
            epochs=config.autoencoder.epochs,
            learning_rate=config.autoencoder.learning_rate,
            validation_fraction=config.autoencoder.validation_fraction,
            mask_probability=config.autoencoder.mask_probability,
            clean_loss_weight=config.autoencoder.clean_loss_weight,
            early_stopping_patience=config.autoencoder.early_stopping_patience,
            min_delta=config.autoencoder.min_delta,
            random_seed=config.autoencoder.random_seed,
        )
        country_key = str(country).lower()
        embeddings_path = config.autoencoder_dir / f"{country_key}_embeddings.parquet"
        reconstruction_path = config.autoencoder_dir / f"{country_key}_reconstruction.parquet"
        masked_reconstruction_path = (
            config.autoencoder_dir / f"{country_key}_masked_reconstruction.parquet"
        )
        metrics_path = config.autoencoder_dir / f"{country_key}_metrics.parquet"
        result.embeddings.to_parquet(embeddings_path, index=False)
        result.reconstruction.to_parquet(reconstruction_path, index=False)
        result.masked_reconstruction.to_parquet(masked_reconstruction_path, index=False)
        result.metrics.to_parquet(metrics_path, index=False)
        output_paths.extend(
            [embeddings_path, reconstruction_path, masked_reconstruction_path, metrics_path]
        )

    return output_paths


class AutoencoderResult:
    def __init__(
        self,
        embeddings: pd.DataFrame,
        reconstruction: pd.DataFrame,
        masked_reconstruction: pd.DataFrame,
        metrics: pd.DataFrame,
    ) -> None:
        self.embeddings = embeddings
        self.reconstruction = reconstruction
        self.masked_reconstruction = masked_reconstruction
        self.metrics = metrics


class MaskedPrediction:
    def __init__(self, reconstructed: NDArray[np.float32], mask: NDArray[np.bool_]) -> None:
        self.reconstructed = reconstructed
        self.mask = mask


def fit_autoencoder_panel(
    panel: pd.DataFrame,
    test_fraction: float,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    validation_fraction: float,
    mask_probability: float,
    clean_loss_weight: float,
    early_stopping_patience: int,
    min_delta: float,
    random_seed: int,
) -> AutoencoderResult:
    """Fit an autoencoder on a chronological train split and reconstruct all dates."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 < mask_probability < 1:
        raise ValueError("mask_probability must be between 0 and 1")
    if clean_loss_weight < 0:
        raise ValueError("clean_loss_weight must be non-negative")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")

    split = _date_ordered_panel_split(panel, test_fraction)
    if split is None:
        raise ValueError("Panel is too short for the requested test split")
    train_validation_panel, test_panel = split
    inner_split = _date_ordered_panel_split(train_validation_panel, validation_fraction)
    if inner_split is None:
        raise ValueError("Training panel is too short for the requested validation split")
    train_panel, validation_panel = inner_split

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.set_num_threads(1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_panel).astype(np.float32)
    x_validation = scaler.transform(validation_panel).astype(np.float32)
    x_all = scaler.transform(panel).astype(np.float32)

    model = CurveAutoencoder(
        input_dim=panel.shape[1] * 2,
        hidden_dim=hidden_dim,
        latent_dim=min(latent_dim, hidden_dim),
        output_dim=panel.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_tensor = torch.from_numpy(x_train)
    validation_tensor = torch.from_numpy(x_validation)
    generator = torch.Generator().manual_seed(random_seed)
    validation_mask = _random_mask(validation_tensor.shape, mask_probability, generator)

    best_state: dict[str, Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    final_train_loss = float("nan")
    final_validation_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_mask = _random_mask(train_tensor.shape, mask_probability, generator)
        train_input = _model_input(train_tensor, train_mask)
        reconstructed = model(train_input)
        masked_loss = loss_fn(reconstructed[train_mask], train_tensor[train_mask])
        clean_reconstructed = model(_model_input(train_tensor, torch.zeros_like(train_mask)))
        clean_loss = loss_fn(clean_reconstructed, train_tensor)
        loss = masked_loss + clean_loss_weight * clean_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_input = _model_input(validation_tensor, validation_mask)
            validation_reconstructed = model(validation_input)
            validation_masked_loss = loss_fn(
                validation_reconstructed[validation_mask],
                validation_tensor[validation_mask],
            )
            validation_clean_reconstructed = model(
                _model_input(validation_tensor, torch.zeros_like(validation_mask))
            )
            validation_clean_loss = loss_fn(validation_clean_reconstructed, validation_tensor)
            validation_loss = validation_masked_loss + clean_loss_weight * validation_clean_loss

        final_train_loss = float(loss.detach().item())
        final_validation_loss = float(validation_loss.detach().item())
        if final_validation_loss < best_validation_loss - min_delta:
            best_validation_loss = final_validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        all_tensor = torch.from_numpy(x_all)
        clean_all_input = _model_input(all_tensor, torch.zeros_like(all_tensor, dtype=torch.bool))
        encoded = model.encoder(clean_all_input).numpy()
        reconstructed_scaled = model(clean_all_input).numpy()
        masked_eval = _masked_prediction(
            model=model,
            values=all_tensor,
            mask_probability=mask_probability,
            random_seed=random_seed + 10_000,
        )

    reconstructed_values = scaler.inverse_transform(reconstructed_scaled)
    masked_reconstructed_values = scaler.inverse_transform(masked_eval.reconstructed)
    split_labels = _split_labels(
        index=panel.index,
        validation_index=validation_panel.index,
        test_index=test_panel.index,
    )
    embeddings = _embedding_frame(
        index=panel.index,
        country=str(panel.attrs.get("country", "")),
        encoded=encoded,
        split_labels=split_labels,
    )
    reconstruction = _reconstruction_frame(
        panel=panel,
        reconstructed=reconstructed_values,
        split_labels=split_labels,
    )
    masked_reconstruction = _reconstruction_frame(
        panel=panel,
        reconstructed=masked_reconstructed_values,
        split_labels=split_labels,
        mask=masked_eval.mask,
    )
    metrics = _metrics_frame(
        reconstruction=pd.concat(
            [
                reconstruction.assign(metric_scope="clean"),
                masked_reconstruction.assign(metric_scope="masked"),
            ],
            ignore_index=True,
        ),
        country=str(panel.attrs.get("country", "")),
        latent_dim=min(latent_dim, hidden_dim),
        hidden_dim=hidden_dim,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        epochs_trained=best_epoch,
        best_validation_loss=best_validation_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
    )
    return AutoencoderResult(
        embeddings=embeddings,
        reconstruction=reconstruction,
        masked_reconstruction=masked_reconstruction,
        metrics=metrics,
    )


def autoencoder_reconstruction_errors(config: ProjectConfig) -> pd.DataFrame:
    """Read autoencoder reconstructions as reconstruction-error rows."""
    if not config.autoencoder_dir.exists():
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for path in sorted(config.autoencoder_dir.glob("*_reconstruction.parquet")):
        if path.name.endswith("_masked_reconstruction.parquet"):
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        test = frame.loc[frame["split"] == "test"].copy()
        if test.empty:
            continue
        test["representation"] = "autoencoder"
        test["n_components"] = config.autoencoder.latent_dim
        test["error"] = test["yield"] - test["fitted_yield"]
        rows.append(test)
    for path in sorted(config.autoencoder_dir.glob("*_masked_reconstruction.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        test = frame.loc[frame["split"] == "test"].copy()
        if test.empty:
            continue
        test["representation"] = "masked_autoencoder"
        test["n_components"] = config.autoencoder.latent_dim
        test["error"] = test["yield"] - test["fitted_yield"]
        rows.append(test)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _date_ordered_panel_split(
    panel: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    split_index = int(len(panel.index) * (1.0 - test_fraction))
    if split_index <= 0 or split_index >= len(panel.index):
        return None
    return panel.iloc[:split_index], panel.iloc[split_index:]


def _split_labels(
    index: pd.Index,
    validation_index: pd.Index,
    test_index: pd.Index,
) -> NDArray[np.str_]:
    validation_dates = set(pd.to_datetime(validation_index))
    test_dates = set(pd.to_datetime(test_index))
    labels = [
        _split_label(date, validation_dates, test_dates)
        for date in pd.to_datetime(index)
    ]
    return np.asarray(labels, dtype=np.str_)


def _split_label(
    date: pd.Timestamp,
    validation_dates: set[pd.Timestamp],
    test_dates: set[pd.Timestamp],
) -> str:
    if date in test_dates:
        return "test"
    if date in validation_dates:
        return "validation"
    return "train"


def _embedding_frame(
    index: pd.Index,
    country: str,
    encoded: NDArray[np.float32],
    split_labels: NDArray[np.str_],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        encoded,
        columns=[f"AE{i}" for i in range(1, encoded.shape[1] + 1)],
    )
    frame.insert(0, "split", split_labels)
    frame.insert(0, "country", country)
    frame.insert(0, "date", pd.to_datetime(index))
    return frame


def _reconstruction_frame(
    panel: pd.DataFrame,
    reconstructed: NDArray[np.float64],
    split_labels: NDArray[np.str_],
    mask: NDArray[np.bool_] | None = None,
) -> pd.DataFrame:
    fitted = pd.DataFrame(reconstructed, index=panel.index, columns=panel.columns)
    actual_long = panel.stack().rename("yield").reset_index()
    fitted_long = fitted.stack().rename("fitted_yield").reset_index()
    actual_long = actual_long.rename(
        columns={actual_long.columns[0]: "date", actual_long.columns[1]: "maturity_years"}
    )
    fitted_long = fitted_long.rename(
        columns={fitted_long.columns[0]: "date", fitted_long.columns[1]: "maturity_years"}
    )
    frame = actual_long.merge(fitted_long, on=["date", "maturity_years"], how="inner")
    frame["country"] = str(panel.attrs.get("country", ""))
    split_by_date = dict(zip(pd.to_datetime(panel.index), split_labels, strict=True))
    frame["split"] = pd.to_datetime(frame["date"]).map(split_by_date)
    if mask is not None:
        mask_long = _mask_frame(panel, mask)
        frame = frame.merge(mask_long, on=["date", "maturity_years"], how="inner")
        frame = frame.loc[frame["is_masked"]].drop(columns=["is_masked"])
    return frame.loc[
        :,
        ["date", "country", "maturity_years", "yield", "fitted_yield", "split"],
    ]


def _mask_frame(panel: pd.DataFrame, mask: NDArray[np.bool_]) -> pd.DataFrame:
    mask_panel = pd.DataFrame(mask, index=panel.index, columns=panel.columns)
    long = mask_panel.stack().rename("is_masked").reset_index()
    return long.rename(columns={long.columns[0]: "date", long.columns[1]: "maturity_years"})


def _metrics_frame(
    reconstruction: pd.DataFrame,
    country: str,
    latent_dim: int,
    hidden_dim: int,
    mask_probability: float,
    clean_loss_weight: float,
    epochs_trained: int,
    best_validation_loss: float,
    final_train_loss: float,
    final_validation_loss: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = reconstruction.copy()
    if "metric_scope" not in frame.columns:
        frame["metric_scope"] = "clean"
    frame["error"] = frame["yield"] - frame["fitted_yield"]
    for group_values, group in frame.groupby(["metric_scope", "split"], sort=True):
        metric_scope, split = group_values
        rows.append(
            {
                "country": country,
                "metric_scope": metric_scope,
                "split": split,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "mask_probability": mask_probability,
                "clean_loss_weight": clean_loss_weight,
                "epochs_trained": epochs_trained,
                "best_validation_loss": best_validation_loss,
                "final_train_loss": final_train_loss,
                "final_validation_loss": final_validation_loss,
                "observations": len(group),
                "dates": group["date"].nunique(),
                "rmse": float(np.sqrt(np.mean(np.square(group["error"])))),
                "mae": float(np.mean(np.abs(group["error"]))),
                "mean_error": float(group["error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _random_mask(
    shape: torch.Size,
    mask_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = torch.rand(shape, generator=generator) < mask_probability
    row_has_mask = mask.any(dim=1)
    if bool(row_has_mask.all()):
        return mask

    missing_rows = torch.where(~row_has_mask)[0]
    columns = torch.randint(
        low=0,
        high=shape[1],
        size=(len(missing_rows),),
        generator=generator,
    )
    mask[missing_rows, columns] = True
    return mask


def _model_input(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.clone()
    masked[mask] = 0.0
    return torch.cat([masked, mask.float()], dim=1)


def _masked_prediction(
    model: CurveAutoencoder,
    values: torch.Tensor,
    mask_probability: float,
    random_seed: int,
) -> MaskedPrediction:
    generator = torch.Generator().manual_seed(random_seed)
    mask = _random_mask(values.shape, mask_probability, generator)
    reconstructed = model(_model_input(values, mask)).numpy()
    return MaskedPrediction(
        reconstructed=reconstructed,
        mask=mask.numpy().astype(np.bool_),
    )
