from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from yieldrep.config import ProjectConfig
from yieldrep.factors.curve import curve_panel
from yieldrep.models.autoencoder import MaskedPrediction


class MaturityTransformer(nn.Module):
    """Small maturity-aware Transformer encoder for masked curve reconstruction."""

    def __init__(
        self,
        n_maturities: int,
        model_dim: int,
        n_heads: int,
        n_layers: int,
        feedforward_dim: int,
        dropout: float,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, model_dim)
        self.maturity_projection = nn.Linear(1, model_dim)
        self.maturity_embedding = nn.Embedding(n_maturities, model_dim)
        self.curve_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_head = nn.Linear(model_dim, 1)
        self.latent_head = nn.Linear(model_dim, latent_dim)

    def forward(self, inputs: Tensor, maturity_features: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, n_maturities, _ = inputs.shape
        maturity_index = torch.arange(n_maturities, device=inputs.device)
        maturity_positions = maturity_features.view(1, n_maturities, 1).to(inputs.device)
        maturity_positions = maturity_positions.expand(batch_size, -1, -1)
        tokens = (
            self.input_projection(inputs)
            + self.maturity_projection(maturity_positions)
            + self.maturity_embedding(maturity_index).unsqueeze(0)
        )
        curve_token = self.curve_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([curve_token, tokens], dim=1))
        curve_encoded = encoded[:, 1:, :]
        reconstructed = self.output_head(curve_encoded).squeeze(-1)
        latent = self.latent_head(encoded[:, 0, :])
        return reconstructed, latent


class TransformerResult:
    def __init__(
        self,
        embeddings: pd.DataFrame,
        reconstruction: pd.DataFrame,
        masked_reconstruction: pd.DataFrame,
        metrics: pd.DataFrame,
        training_history: pd.DataFrame,
    ) -> None:
        self.embeddings = embeddings
        self.reconstruction = reconstruction
        self.masked_reconstruction = masked_reconstruction
        self.metrics = metrics
        self.training_history = training_history


def build_transformer(config: ProjectConfig) -> list[Path]:
    """Train per-country maturity Transformers and write reconstruction artifacts."""
    curves = pd.read_parquet(config.curves_path)
    output_paths: list[Path] = []
    config.transformer_dir.mkdir(parents=True, exist_ok=True)

    for country in sorted(curves["country"].dropna().unique()):
        panel = curve_panel(curves, str(country)).ffill().dropna()
        panel.attrs["country"] = str(country)
        if panel.shape[1] < config.pca.min_maturities:
            continue
        split = _date_ordered_panel_split(panel, config.evaluation.test_fraction)
        if split is None:
            continue
        train_panel, _ = split
        if len(train_panel) < config.transformer.min_train_dates:
            continue

        result = fit_transformer_panel(
            panel=panel,
            test_fraction=config.evaluation.test_fraction,
            latent_dim=config.transformer.latent_dim,
            model_dim=config.transformer.model_dim,
            n_heads=config.transformer.n_heads,
            n_layers=config.transformer.n_layers,
            feedforward_dim=config.transformer.feedforward_dim,
            dropout=config.transformer.dropout,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            weight_decay=config.transformer.weight_decay,
            validation_fraction=config.transformer.validation_fraction,
            mask_probability=config.transformer.mask_probability,
            clean_loss_weight=config.transformer.clean_loss_weight,
            early_stopping_patience=config.transformer.early_stopping_patience,
            min_delta=config.transformer.min_delta,
            random_seed=config.transformer.random_seed,
            max_train_dates=config.transformer.max_train_dates,
        )
        country_key = str(country).lower()
        embeddings_path = config.transformer_dir / f"{country_key}_embeddings.parquet"
        reconstruction_path = config.transformer_dir / f"{country_key}_reconstruction.parquet"
        masked_path = config.transformer_dir / f"{country_key}_masked_reconstruction.parquet"
        metrics_path = config.transformer_dir / f"{country_key}_metrics.parquet"
        history_path = config.transformer_dir / f"{country_key}_training_history.parquet"
        result.embeddings.to_parquet(embeddings_path, index=False)
        result.reconstruction.to_parquet(reconstruction_path, index=False)
        result.masked_reconstruction.to_parquet(masked_path, index=False)
        result.metrics.to_parquet(metrics_path, index=False)
        result.training_history.to_parquet(history_path, index=False)
        output_paths.extend(
            [embeddings_path, reconstruction_path, masked_path, metrics_path, history_path]
        )

    return output_paths


def fit_transformer_panel(
    panel: pd.DataFrame,
    test_fraction: float,
    latent_dim: int,
    model_dim: int,
    n_heads: int,
    n_layers: int,
    feedforward_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    mask_probability: float,
    clean_loss_weight: float,
    early_stopping_patience: int,
    min_delta: float,
    random_seed: int,
    max_train_dates: int | None = None,
) -> TransformerResult:
    _validate_hyperparameters(
        latent_dim=latent_dim,
        model_dim=model_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_fraction=validation_fraction,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
        max_train_dates=max_train_dates,
    )
    split = _date_ordered_panel_split(panel, test_fraction)
    if split is None:
        raise ValueError("Panel is too short for the requested test split")
    train_validation_panel, test_panel = split
    inner_split = _date_ordered_panel_split(train_validation_panel, validation_fraction)
    if inner_split is None:
        raise ValueError("Training panel is too short for the requested validation split")
    train_panel, validation_panel = inner_split
    train_panel = _cap_training_panel(train_panel, max_train_dates)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.set_num_threads(1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_panel).astype(np.float32)
    x_validation = scaler.transform(validation_panel).astype(np.float32)
    x_all = scaler.transform(panel).astype(np.float32)
    maturity_features = _maturity_features(panel.columns.astype(float).to_numpy())

    model = MaturityTransformer(
        n_maturities=x_train.shape[1],
        model_dim=model_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        latent_dim=latent_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_tensor = torch.from_numpy(x_train)
    validation_tensor = torch.from_numpy(x_validation)
    maturity_tensor = torch.from_numpy(maturity_features)
    generator = torch.Generator().manual_seed(random_seed)
    validation_mask = _random_mask(validation_tensor.shape, mask_probability, generator)

    best_state: dict[str, Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    final_train_loss = float("nan")
    final_validation_loss = float("nan")
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = _empty_loss_totals()
        for batch in _batch_iterator(train_tensor, batch_size, generator):
            optimizer.zero_grad()
            train_mask = _random_mask(batch.shape, mask_probability, generator)
            reconstructed, _ = model(_model_input(batch, train_mask), maturity_tensor)
            masked_loss = loss_fn(reconstructed[train_mask], batch[train_mask])
            clean_reconstructed, _ = model(
                _model_input(batch, torch.zeros_like(train_mask)),
                maturity_tensor,
            )
            clean_loss = loss_fn(clean_reconstructed, batch)
            loss = masked_loss + clean_loss_weight * clean_loss
            loss.backward()
            optimizer.step()
            _update_loss_totals(totals, batch.shape[0], loss, masked_loss, clean_loss)

        model.eval()
        with torch.no_grad():
            validation_input = _model_input(validation_tensor, validation_mask)
            validation_reconstructed, _ = model(validation_input, maturity_tensor)
            validation_masked_loss = loss_fn(
                validation_reconstructed[validation_mask],
                validation_tensor[validation_mask],
            )
            validation_clean_reconstructed, _ = model(
                _model_input(validation_tensor, torch.zeros_like(validation_mask)),
                maturity_tensor,
            )
            validation_clean_loss = loss_fn(validation_clean_reconstructed, validation_tensor)
            validation_loss = validation_masked_loss + clean_loss_weight * validation_clean_loss

        final_train_loss = totals["loss"] / totals["rows"]
        final_validation_loss = float(validation_loss.detach().item())
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": final_train_loss,
                "train_masked_loss": totals["masked_loss"] / totals["rows"],
                "train_clean_loss": totals["clean_loss"] / totals["rows"],
                "validation_loss": final_validation_loss,
                "validation_masked_loss": float(validation_masked_loss.detach().item()),
                "validation_clean_loss": float(validation_clean_loss.detach().item()),
            }
        )
        if final_validation_loss < best_validation_loss - min_delta:
            best_validation_loss = final_validation_loss
            best_epoch = epoch
            best_state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
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
        clean_input = _model_input(all_tensor, torch.zeros_like(all_tensor, dtype=torch.bool))
        reconstructed_scaled, encoded = model(clean_input, maturity_tensor)
        masked_eval = _masked_prediction(
            model=model,
            values=all_tensor,
            maturity_features=maturity_tensor,
            mask_probability=mask_probability,
            random_seed=random_seed + 20_000,
        )

    reconstructed = scaler.inverse_transform(reconstructed_scaled.numpy())
    masked_reconstructed = scaler.inverse_transform(masked_eval.reconstructed)
    split_labels = _split_labels(panel.index, validation_panel.index, test_panel.index)
    clean_frame = _reconstruction_frame(panel, reconstructed, split_labels)
    masked_frame = _reconstruction_frame(panel, masked_reconstructed, split_labels, masked_eval.mask)
    metrics = _metrics_frame(
        reconstruction=pd.concat(
            [clean_frame.assign(metric_scope="clean"), masked_frame.assign(metric_scope="masked")],
            ignore_index=True,
        ),
        country=str(panel.attrs.get("country", "")),
        latent_dim=latent_dim,
        model_dim=model_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        batch_size=batch_size,
        weight_decay=weight_decay,
        mask_probability=mask_probability,
        clean_loss_weight=clean_loss_weight,
        epochs_trained=best_epoch,
        best_validation_loss=best_validation_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
    )
    return TransformerResult(
        embeddings=_embedding_frame(panel.index, str(panel.attrs.get("country", "")), encoded.numpy(), split_labels),
        reconstruction=clean_frame,
        masked_reconstruction=masked_frame,
        metrics=metrics,
        training_history=_training_history_frame(history_rows, str(panel.attrs.get("country", "")), best_epoch),
    )


def transformer_reconstruction_errors(config: ProjectConfig) -> pd.DataFrame:
    """Read Transformer reconstructions as reconstruction-error rows."""
    if not config.transformer_dir.exists():
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for path in sorted(config.transformer_dir.glob("*_reconstruction.parquet")):
        if path.name.endswith("_masked_reconstruction.parquet"):
            continue
        frame = pd.read_parquet(path)
        test = frame.loc[frame["split"] == "test"].copy()
        if test.empty:
            continue
        test["reconstruction_task"] = "clean_reconstruction"
        test["representation"] = "transformer"
        test["n_components"] = config.transformer.latent_dim
        test["error"] = test["yield"] - test["fitted_yield"]
        rows.append(test)
    for path in sorted(config.transformer_dir.glob("*_masked_reconstruction.parquet")):
        frame = pd.read_parquet(path)
        test = frame.loc[frame["split"] == "test"].copy()
        if test.empty:
            continue
        test["reconstruction_task"] = "masked_maturity_reconstruction"
        test["representation"] = "masked_transformer"
        test["n_components"] = config.transformer.latent_dim
        test["error"] = test["yield"] - test["fitted_yield"]
        rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _validate_hyperparameters(
    latent_dim: int,
    model_dim: int,
    n_heads: int,
    n_layers: int,
    feedforward_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    mask_probability: float,
    clean_loss_weight: float,
    early_stopping_patience: int,
    min_delta: float,
    max_train_dates: int | None,
) -> None:
    if latent_dim <= 0 or model_dim <= 0 or n_heads <= 0 or n_layers <= 0 or feedforward_dim <= 0:
        raise ValueError("Transformer dimensions must be positive")
    if model_dim % n_heads != 0:
        raise ValueError("model_dim must be divisible by n_heads")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if not 0 < validation_fraction < 1 or not 0 < mask_probability < 1:
        raise ValueError("Fractions must be between 0 and 1")
    if clean_loss_weight < 0 or early_stopping_patience <= 0 or min_delta < 0:
        raise ValueError("Invalid training hyperparameters")
    if max_train_dates is not None and max_train_dates <= 0:
        raise ValueError("max_train_dates must be positive when provided")


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


def _cap_training_panel(panel: pd.DataFrame, max_train_dates: int | None) -> pd.DataFrame:
    if max_train_dates is None or len(panel.index) <= max_train_dates:
        return panel
    positions = np.linspace(0, len(panel.index) - 1, max_train_dates).round().astype(int)
    return panel.iloc[np.unique(positions)]


def _maturity_features(maturities: NDArray[np.float64]) -> NDArray[np.float32]:
    log_maturities = np.log1p(maturities)
    denominator = log_maturities.max() - log_maturities.min()
    if denominator <= 0:
        scaled = np.zeros_like(log_maturities)
    else:
        scaled = (log_maturities - log_maturities.min()) / denominator
    return scaled.astype(np.float32)


def _model_input(values: Tensor, mask: Tensor) -> Tensor:
    masked = values.clone()
    masked[mask] = 0.0
    return torch.stack([masked, mask.float()], dim=-1)


def _random_mask(shape: torch.Size, mask_probability: float, generator: torch.Generator) -> Tensor:
    mask = torch.rand(shape, generator=generator) < mask_probability
    row_has_mask = mask.any(dim=1)
    if bool(row_has_mask.all()):
        return mask
    missing_rows = torch.where(~row_has_mask)[0]
    columns = torch.randint(0, shape[1], (len(missing_rows),), generator=generator)
    mask[missing_rows, columns] = True
    return mask


def _batch_iterator(values: Tensor, batch_size: int, generator: torch.Generator) -> list[Tensor]:
    permutation = torch.randperm(values.shape[0], generator=generator)
    return [values[permutation[start : start + batch_size]] for start in range(0, values.shape[0], batch_size)]


def _empty_loss_totals() -> dict[str, float]:
    return {"rows": 0.0, "loss": 0.0, "masked_loss": 0.0, "clean_loss": 0.0}


def _update_loss_totals(
    totals: dict[str, float],
    rows: int,
    loss: Tensor,
    masked_loss: Tensor,
    clean_loss: Tensor,
) -> None:
    totals["rows"] += rows
    totals["loss"] += float(loss.detach().item()) * rows
    totals["masked_loss"] += float(masked_loss.detach().item()) * rows
    totals["clean_loss"] += float(clean_loss.detach().item()) * rows


def _masked_prediction(
    model: MaturityTransformer,
    values: Tensor,
    maturity_features: Tensor,
    mask_probability: float,
    random_seed: int,
) -> MaskedPrediction:
    generator = torch.Generator().manual_seed(random_seed)
    mask = _random_mask(values.shape, mask_probability, generator)
    reconstructed, _ = model(_model_input(values, mask), maturity_features)
    return MaskedPrediction(reconstructed.numpy(), mask.numpy().astype(np.bool_))


def _split_labels(
    index: pd.Index,
    validation_index: pd.Index,
    test_index: pd.Index,
) -> NDArray[np.str_]:
    validation_dates = set(pd.to_datetime(validation_index))
    test_dates = set(pd.to_datetime(test_index))
    labels = [
        "test" if date in test_dates else "validation" if date in validation_dates else "train"
        for date in pd.to_datetime(index)
    ]
    return np.asarray(labels, dtype=np.str_)


def _embedding_frame(
    index: pd.Index,
    country: str,
    encoded: NDArray[np.float32],
    split_labels: NDArray[np.str_],
) -> pd.DataFrame:
    frame = pd.DataFrame(encoded, columns=[f"TE{i}" for i in range(1, encoded.shape[1] + 1)])
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
    actual_long = actual_long.rename(columns={actual_long.columns[0]: "date", actual_long.columns[1]: "maturity_years"})
    fitted_long = fitted_long.rename(columns={fitted_long.columns[0]: "date", fitted_long.columns[1]: "maturity_years"})
    frame = actual_long.merge(fitted_long, on=["date", "maturity_years"], how="inner")
    frame["country"] = str(panel.attrs.get("country", ""))
    frame["split"] = pd.to_datetime(frame["date"]).map(dict(zip(pd.to_datetime(panel.index), split_labels, strict=True)))
    if mask is not None:
        mask_frame = pd.DataFrame(mask, index=panel.index, columns=panel.columns)
        mask_long = mask_frame.stack().rename("is_masked").reset_index()
        mask_long = mask_long.rename(columns={mask_long.columns[0]: "date", mask_long.columns[1]: "maturity_years"})
        frame = frame.merge(mask_long, on=["date", "maturity_years"], how="inner")
        frame = frame.loc[frame["is_masked"]].drop(columns=["is_masked"])
    return frame.loc[:, ["date", "country", "maturity_years", "yield", "fitted_yield", "split"]]


def _metrics_frame(
    reconstruction: pd.DataFrame,
    country: str,
    latent_dim: int,
    model_dim: int,
    n_heads: int,
    n_layers: int,
    feedforward_dim: int,
    dropout: float,
    batch_size: int,
    weight_decay: float,
    mask_probability: float,
    clean_loss_weight: float,
    epochs_trained: int,
    best_validation_loss: float,
    final_train_loss: float,
    final_validation_loss: float,
) -> pd.DataFrame:
    frame = reconstruction.copy()
    frame["error"] = frame["yield"] - frame["fitted_yield"]
    rows: list[dict[str, object]] = []
    for (metric_scope, split), group in frame.groupby(["metric_scope", "split"], sort=True):
        rows.append(
            {
                "country": country,
                "metric_scope": metric_scope,
                "split": split,
                "latent_dim": latent_dim,
                "model_dim": model_dim,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "feedforward_dim": feedforward_dim,
                "dropout": dropout,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
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


def _training_history_frame(
    history_rows: list[dict[str, object]],
    country: str,
    best_epoch: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(history_rows)
    if frame.empty:
        return pd.DataFrame()
    frame.insert(0, "country", country)
    frame["is_best_epoch"] = frame["epoch"].eq(best_epoch)
    return frame
