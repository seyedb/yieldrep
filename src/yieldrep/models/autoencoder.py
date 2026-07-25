from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import nn

from yieldrep.config import ProjectConfig
from yieldrep.factors.curve import curve_panel


class CurveAutoencoder(nn.Module):
    """Small MLP autoencoder for curve reconstruction, not forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
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
            random_seed=config.autoencoder.random_seed,
        )
        country_key = str(country).lower()
        embeddings_path = config.autoencoder_dir / f"{country_key}_embeddings.parquet"
        reconstruction_path = config.autoencoder_dir / f"{country_key}_reconstruction.parquet"
        result.embeddings.to_parquet(embeddings_path, index=False)
        result.reconstruction.to_parquet(reconstruction_path, index=False)
        output_paths.extend([embeddings_path, reconstruction_path])

    return output_paths


class AutoencoderResult:
    def __init__(self, embeddings: pd.DataFrame, reconstruction: pd.DataFrame) -> None:
        self.embeddings = embeddings
        self.reconstruction = reconstruction


def fit_autoencoder_panel(
    panel: pd.DataFrame,
    test_fraction: float,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
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

    split = _date_ordered_panel_split(panel, test_fraction)
    if split is None:
        raise ValueError("Panel is too short for the requested test split")
    train_panel, test_panel = split

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.set_num_threads(1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_panel).astype(np.float32)
    x_all = scaler.transform(panel).astype(np.float32)

    model = CurveAutoencoder(
        input_dim=panel.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=min(latent_dim, hidden_dim),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_tensor = torch.from_numpy(x_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(train_tensor)
        loss = loss_fn(reconstructed, train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        all_tensor = torch.from_numpy(x_all)
        encoded = model.encoder(all_tensor).numpy()
        reconstructed_scaled = model(all_tensor).numpy()

    reconstructed_values = scaler.inverse_transform(reconstructed_scaled)
    split_labels = _split_labels(panel.index, test_panel.index)
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
    return AutoencoderResult(embeddings=embeddings, reconstruction=reconstruction)


def autoencoder_reconstruction_errors(config: ProjectConfig) -> pd.DataFrame:
    """Read autoencoder reconstructions as reconstruction-error rows."""
    if not config.autoencoder_dir.exists():
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for path in sorted(config.autoencoder_dir.glob("*_reconstruction.parquet")):
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


def _split_labels(index: pd.Index, test_index: pd.Index) -> NDArray[np.str_]:
    test_dates = set(pd.to_datetime(test_index))
    labels = ["test" if date in test_dates else "train" for date in pd.to_datetime(index)]
    return np.asarray(labels, dtype=np.str_)


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
    return frame.loc[
        :,
        ["date", "country", "maturity_years", "yield", "fitted_yield", "split"],
    ]
