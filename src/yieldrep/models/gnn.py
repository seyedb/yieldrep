from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from yieldrep.config import ProjectConfig
from yieldrep.factors.carry import CARRY_ROLL_FEATURE_COLUMNS
from yieldrep.graph.dataset import build_maturity_graph_dataset
from yieldrep.models.autoencoder import MaskedPrediction

NODE_FEATURE_COLUMNS = [
    "yield",
    "yield_change_1d",
    "realized_vol",
    *CARRY_ROLL_FEATURE_COLUMNS,
]

GNN_EDGE_ABLATION_COLUMNS = [
    "country",
    "edge_mode",
    "edge_count",
    "masked_test_rmse",
    "masked_test_mae",
    "clean_test_rmse",
    "clean_test_mae",
    "epochs_trained",
    "best_validation_loss",
    "masked_rmse_vs_adjacent_only",
]

EdgeMode = Literal["adjacent", "adjacent_correlation"]


class MaturityGraphAutoencoder(nn.Module):
    """Small GCN-style autoencoder for masked maturity reconstruction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
        adjacency: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_dim, 1)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = torch.nn.functional.gelu(self.input_projection(inputs))
        for layer in self.graph_layers:
            hidden = torch.einsum("ij,bjf->bif", self.adjacency, hidden)
            hidden = torch.nn.functional.gelu(layer(hidden))
            hidden = self.dropout(hidden)
        reconstructed = self.output_head(hidden).squeeze(-1)
        latent = self.latent_head(hidden.mean(dim=1))
        return reconstructed, latent


class GNNResult:
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


def build_gnn(config: ProjectConfig) -> list[Path]:
    """Train per-country maturity-graph autoencoders and write reconstruction artifacts."""
    if not config.graph_nodes_path.exists() or not config.graph_edges_path.exists():
        build_maturity_graph_dataset(config)

    nodes = pd.read_parquet(config.graph_nodes_path)
    edges = pd.read_parquet(config.graph_edges_path)
    output_paths: list[Path] = []
    config.gnn_dir.mkdir(parents=True, exist_ok=True)

    for country in sorted(nodes["country"].dropna().unique()):
        country_nodes = nodes.loc[nodes["country"] == country].copy()
        panel = _feature_panel(country_nodes)
        if panel.shape[1] < config.pca.min_maturities:
            continue
        split = _date_ordered_split(panel, config.evaluation.test_fraction)
        if split is None:
            continue
        train_panel, _ = split
        if len(train_panel) < config.gnn.min_train_dates:
            continue

        country_edges = _filter_edges(
            edges.loc[edges["country"] == country].copy(),
            config.gnn.edge_mode,
        )
        result = fit_gnn_panel(
            panel=panel,
            edges=country_edges,
            test_fraction=config.evaluation.test_fraction,
            latent_dim=config.gnn.latent_dim,
            hidden_dim=config.gnn.hidden_dim,
            n_layers=config.gnn.n_layers,
            dropout=config.gnn.dropout,
            epochs=config.gnn.epochs,
            batch_size=config.gnn.batch_size,
            learning_rate=config.gnn.learning_rate,
            weight_decay=config.gnn.weight_decay,
            validation_fraction=config.gnn.validation_fraction,
            mask_probability=config.gnn.mask_probability,
            clean_loss_weight=config.gnn.clean_loss_weight,
            early_stopping_patience=config.gnn.early_stopping_patience,
            min_delta=config.gnn.min_delta,
            random_seed=config.gnn.random_seed,
            max_train_dates=config.gnn.max_train_dates,
        )

        country_key = str(country).lower()
        embeddings_path = config.gnn_dir / f"{country_key}_embeddings.parquet"
        reconstruction_path = config.gnn_dir / f"{country_key}_reconstruction.parquet"
        masked_path = config.gnn_dir / f"{country_key}_masked_reconstruction.parquet"
        metrics_path = config.gnn_dir / f"{country_key}_metrics.parquet"
        history_path = config.gnn_dir / f"{country_key}_training_history.parquet"
        result.embeddings.to_parquet(embeddings_path, index=False)
        result.reconstruction.to_parquet(reconstruction_path, index=False)
        result.masked_reconstruction.to_parquet(masked_path, index=False)
        result.metrics.to_parquet(metrics_path, index=False)
        result.training_history.to_parquet(history_path, index=False)
        output_paths.extend(
            [embeddings_path, reconstruction_path, masked_path, metrics_path, history_path]
        )

    return output_paths


def build_gnn_edge_ablation(config: ProjectConfig) -> Path:
    """Compare adjacent-only and adjacent+correlation graph edges.

    This ablation tests whether the extra correlation edges improve the
    self-supervised masked maturity reconstruction task. It writes only a
    summary table and does not overwrite the canonical graph-AE artifacts.
    """
    if not config.graph_nodes_path.exists() or not config.graph_edges_path.exists():
        build_maturity_graph_dataset(config)

    nodes = pd.read_parquet(config.graph_nodes_path)
    edges = pd.read_parquet(config.graph_edges_path)
    rows: list[dict[str, object]] = []

    for country in sorted(nodes["country"].dropna().unique()):
        country_nodes = nodes.loc[nodes["country"] == country].copy()
        panel = _feature_panel(country_nodes)
        if panel.shape[1] < config.pca.min_maturities:
            continue
        split = _date_ordered_split(panel, config.evaluation.test_fraction)
        if split is None:
            continue
        train_panel, _ = split
        if len(train_panel) < config.gnn.min_train_dates:
            continue

        country_edges = edges.loc[edges["country"] == country].copy()
        edge_modes: list[EdgeMode] = ["adjacent", "adjacent_correlation"]
        for edge_mode in edge_modes:
            mode_edges = _filter_edges(country_edges, edge_mode)
            result = fit_gnn_panel(
                panel=panel,
                edges=mode_edges,
                test_fraction=config.evaluation.test_fraction,
                latent_dim=config.gnn.latent_dim,
                hidden_dim=config.gnn.hidden_dim,
                n_layers=config.gnn.n_layers,
                dropout=config.gnn.dropout,
                epochs=config.gnn.epochs,
                batch_size=config.gnn.batch_size,
                learning_rate=config.gnn.learning_rate,
                weight_decay=config.gnn.weight_decay,
                validation_fraction=config.gnn.validation_fraction,
                mask_probability=config.gnn.mask_probability,
                clean_loss_weight=config.gnn.clean_loss_weight,
                early_stopping_patience=config.gnn.early_stopping_patience,
                min_delta=config.gnn.min_delta,
                random_seed=config.gnn.random_seed,
                max_train_dates=config.gnn.max_train_dates,
            )
            rows.append(_gnn_ablation_row(str(country), edge_mode, mode_edges, result.metrics))

    table = pd.DataFrame(rows, columns=GNN_EDGE_ABLATION_COLUMNS)
    if not table.empty:
        table = _add_adjacent_only_comparison(table)

    config.tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.gnn_edge_ablation_table_path, index=False)
    return config.gnn_edge_ablation_table_path


def fit_gnn_panel(
    panel: pd.DataFrame,
    edges: pd.DataFrame,
    test_fraction: float,
    latent_dim: int,
    hidden_dim: int,
    n_layers: int,
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
    max_train_dates: int | None,
) -> GNNResult:
    _validate_hyperparameters(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
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

    split = _date_ordered_split(panel, test_fraction)
    if split is None:
        raise ValueError("Panel is too short for the requested test split")
    train_validation_panel, test_panel = split
    inner_split = _date_ordered_split(train_validation_panel, validation_fraction)
    if inner_split is None:
        raise ValueError("Training panel is too short for the requested validation split")
    train_panel, validation_panel = inner_split
    train_panel = _cap_training_panel(train_panel, max_train_dates)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.set_num_threads(1)

    maturities = _panel_maturities(panel)
    x_train, y_train, feature_scaler, yield_scaler = _fit_scaled_arrays(train_panel)
    x_validation = _transform_feature_array(validation_panel, feature_scaler)
    y_validation = _transform_yield_array(validation_panel, yield_scaler)
    x_all = _transform_feature_array(panel, feature_scaler)

    adjacency = torch.from_numpy(_normalized_adjacency(maturities, edges))
    model = MaturityGraphAutoencoder(
        input_dim=x_train.shape[-1] + 1,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        dropout=dropout,
        adjacency=adjacency,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_features = torch.from_numpy(x_train)
    train_targets = torch.from_numpy(y_train)
    validation_features = torch.from_numpy(x_validation)
    validation_targets = torch.from_numpy(y_validation)
    generator = torch.Generator().manual_seed(random_seed)
    validation_mask = _random_mask(validation_targets.shape, mask_probability, generator)

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
        for batch_features, batch_targets in _batch_iterator(
            train_features, train_targets, batch_size, generator
        ):
            optimizer.zero_grad()
            train_mask = _random_mask(batch_targets.shape, mask_probability, generator)
            reconstructed, _ = model(_model_input(batch_features, train_mask))
            masked_loss = loss_fn(reconstructed[train_mask], batch_targets[train_mask])
            clean_reconstructed, _ = model(
                _model_input(batch_features, torch.zeros_like(train_mask))
            )
            clean_loss = loss_fn(clean_reconstructed, batch_targets)
            loss = masked_loss + clean_loss_weight * clean_loss
            loss.backward()
            optimizer.step()
            _update_loss_totals(totals, batch_targets.shape[0], loss, masked_loss, clean_loss)

        model.eval()
        with torch.no_grad():
            validation_reconstructed, _ = model(_model_input(validation_features, validation_mask))
            validation_masked_loss = loss_fn(
                validation_reconstructed[validation_mask],
                validation_targets[validation_mask],
            )
            validation_clean_reconstructed, _ = model(
                _model_input(validation_features, torch.zeros_like(validation_mask))
            )
            validation_clean_loss = loss_fn(validation_clean_reconstructed, validation_targets)
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
            best_state = {
                name: parameter.detach().clone() for name, parameter in model.state_dict().items()
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
        all_features = torch.from_numpy(x_all)
        clean_input = _model_input(
            all_features,
            torch.zeros(all_features.shape[:2], dtype=torch.bool),
        )
        reconstructed_scaled, encoded = model(clean_input)
        masked_eval = _masked_prediction(
            model=model,
            values=all_features,
            target_shape=torch.Size(reconstructed_scaled.shape),
            mask_probability=mask_probability,
            random_seed=random_seed + 30_000,
        )

    reconstructed = _inverse_yield_array(reconstructed_scaled.numpy(), yield_scaler)
    masked_reconstructed = _inverse_yield_array(masked_eval.reconstructed, yield_scaler)
    yield_panel = panel.xs("yield", axis=1, level="feature")
    split_labels = _split_labels(panel.index, validation_panel.index, test_panel.index)
    clean_frame = _reconstruction_frame(yield_panel, reconstructed, split_labels)
    masked_frame = _reconstruction_frame(
        yield_panel, masked_reconstructed, split_labels, masked_eval.mask
    )
    metrics = _metrics_frame(
        reconstruction=pd.concat(
            [
                clean_frame.assign(metric_scope="clean"),
                masked_frame.assign(metric_scope="masked"),
            ],
            ignore_index=True,
        ),
        country=str(panel.attrs.get("country", "")),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
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
    return GNNResult(
        embeddings=_embedding_frame(
            panel.index, str(panel.attrs.get("country", "")), encoded.numpy(), split_labels
        ),
        reconstruction=clean_frame,
        masked_reconstruction=masked_frame,
        metrics=metrics,
        training_history=_training_history_frame(
            history_rows, str(panel.attrs.get("country", "")), best_epoch
        ),
    )


def gnn_reconstruction_errors(config: ProjectConfig) -> pd.DataFrame:
    """Read graph autoencoder reconstructions as reconstruction-error rows."""
    if not config.gnn_dir.exists():
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for path in sorted(config.gnn_dir.glob("*_reconstruction.parquet")):
        if path.name.endswith("_masked_reconstruction.parquet"):
            continue
        frame = pd.read_parquet(path)
        test = frame.loc[frame["split"] == "test"].copy()
        if test.empty:
            continue
        test["reconstruction_task"] = "clean_reconstruction"
        test["representation"] = "graph_autoencoder"
        test["n_components"] = config.gnn.latent_dim
        test["error"] = test["yield"] - test["fitted_yield"]
        rows.append(test)

    for path in sorted(config.gnn_dir.glob("*_masked_reconstruction.parquet")):
        frame = pd.read_parquet(path)
        test = frame.loc[frame["split"] == "test"].copy()
        if test.empty:
            continue
        test["reconstruction_task"] = "masked_maturity_reconstruction"
        test["representation"] = "masked_graph_autoencoder"
        test["n_components"] = config.gnn.latent_dim
        test["error"] = test["yield"] - test["fitted_yield"]
        rows.append(test)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _filter_edges(edges: pd.DataFrame, edge_mode: EdgeMode) -> pd.DataFrame:
    if edge_mode == "adjacent":
        return edges.loc[edges["edge_type"] == "adjacent"].copy()
    if edge_mode == "adjacent_correlation":
        return edges.loc[edges["edge_type"].isin(["adjacent", "correlation"])].copy()
    raise ValueError(f"Unsupported edge mode: {edge_mode}")


def _gnn_ablation_row(
    country: str,
    edge_mode: EdgeMode,
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, object]:
    masked_test = _metric_scope_split(metrics, "masked", "test")
    clean_test = _metric_scope_split(metrics, "clean", "test")
    metadata = metrics.iloc[0] if not metrics.empty else pd.Series(dtype=object)
    return {
        "country": country,
        "edge_mode": edge_mode,
        "edge_count": len(edges),
        "masked_test_rmse": _metric_value(masked_test, "rmse"),
        "masked_test_mae": _metric_value(masked_test, "mae"),
        "clean_test_rmse": _metric_value(clean_test, "rmse"),
        "clean_test_mae": _metric_value(clean_test, "mae"),
        "epochs_trained": _metric_value(metadata, "epochs_trained"),
        "best_validation_loss": _metric_value(metadata, "best_validation_loss"),
        "masked_rmse_vs_adjacent_only": np.nan,
    }


def _metric_scope_split(metrics: pd.DataFrame, metric_scope: str, split: str) -> pd.Series:
    selected = metrics.loc[(metrics["metric_scope"] == metric_scope) & (metrics["split"] == split)]
    return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)


def _metric_value(row: pd.Series, column: str) -> float:
    return float(row[column]) if column in row.index and pd.notna(row[column]) else float("nan")


def _add_adjacent_only_comparison(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    adjacent = (
        frame.loc[frame["edge_mode"] == "adjacent", ["country", "masked_test_rmse"]]
        .rename(columns={"masked_test_rmse": "adjacent_only_masked_test_rmse"})
        .copy()
    )
    frame = frame.merge(adjacent, on="country", how="left")
    frame["masked_rmse_vs_adjacent_only"] = (
        frame["masked_test_rmse"] - frame["adjacent_only_masked_test_rmse"]
    )
    return (
        frame.loc[:, GNN_EDGE_ABLATION_COLUMNS]
        .sort_values(["country", "edge_mode"])
        .reset_index(drop=True)
    )


def _feature_panel(nodes: pd.DataFrame) -> pd.DataFrame:
    frame = nodes.loc[:, ["date", "country", "maturity_years", *NODE_FEATURE_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame[NODE_FEATURE_COLUMNS] = frame[NODE_FEATURE_COLUMNS].astype(float)
    country = str(frame["country"].dropna().iloc[0])
    panel = frame.pivot(index="date", columns="maturity_years", values=NODE_FEATURE_COLUMNS)
    panel.columns.names = ["feature", "maturity_years"]
    panel = panel.sort_index(axis=0).sort_index(axis=1, level="maturity_years")
    panel = panel.ffill().dropna(
        subset=[("yield", maturity) for maturity in _panel_maturities(panel)]
    )
    panel = panel.fillna(0.0)
    panel.attrs["country"] = country
    return panel


def _fit_scaled_arrays(
    panel: pd.DataFrame,
) -> tuple[NDArray[np.float32], NDArray[np.float32], StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler()
    yield_scaler = StandardScaler()
    features = _panel_to_feature_array(panel)
    yields = _panel_to_yield_array(panel)
    x_scaled = feature_scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(
        features.shape
    )
    y_scaled = yield_scaler.fit_transform(yields.reshape(-1, 1)).reshape(yields.shape)
    return (
        x_scaled.astype(np.float32),
        y_scaled.astype(np.float32),
        feature_scaler,
        yield_scaler,
    )


def _transform_feature_array(panel: pd.DataFrame, scaler: StandardScaler) -> NDArray[np.float32]:
    features = _panel_to_feature_array(panel)
    scaled = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    return cast(NDArray[np.float32], scaled.astype(np.float32))


def _transform_yield_array(panel: pd.DataFrame, scaler: StandardScaler) -> NDArray[np.float32]:
    yields = _panel_to_yield_array(panel)
    scaled = scaler.transform(yields.reshape(-1, 1)).reshape(yields.shape)
    return cast(NDArray[np.float32], scaled.astype(np.float32))


def _inverse_yield_array(
    values: NDArray[np.float32], scaler: StandardScaler
) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape),
    )


def _panel_to_feature_array(panel: pd.DataFrame) -> NDArray[np.float64]:
    values = np.stack(
        [
            panel.xs(feature, axis=1, level="feature").to_numpy(dtype=float)
            for feature in NODE_FEATURE_COLUMNS
        ],
        axis=-1,
    )
    return cast(NDArray[np.float64], values.astype(np.float64))


def _panel_to_yield_array(panel: pd.DataFrame) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        panel.xs("yield", axis=1, level="feature").to_numpy(dtype=float),
    )


def _normalized_adjacency(
    maturities: NDArray[np.float64], edges: pd.DataFrame
) -> NDArray[np.float32]:
    index_by_maturity = {float(maturity): index for index, maturity in enumerate(maturities)}
    adjacency = np.eye(len(maturities), dtype=np.float32)
    for row in edges.itertuples(index=False):
        source = float(row.source_maturity_years)
        target = float(row.target_maturity_years)
        if source not in index_by_maturity or target not in index_by_maturity:
            continue
        adjacency[index_by_maturity[target], index_by_maturity[source]] += float(row.edge_weight)

    degree = adjacency.sum(axis=1)
    degree[degree <= 0.0] = 1.0
    return cast(NDArray[np.float32], (adjacency / degree[:, None]).astype(np.float32))


def _panel_maturities(panel: pd.DataFrame) -> NDArray[np.float64]:
    maturities = panel.columns.get_level_values("maturity_years").unique().to_numpy(dtype=float)
    return np.sort(maturities)


def _model_input(features: Tensor, mask: Tensor) -> Tensor:
    inputs = features.clone()
    inputs[:, :, 0][mask] = 0.0
    return torch.cat([inputs, mask.unsqueeze(-1).float()], dim=-1)


def _random_mask(shape: torch.Size, mask_probability: float, generator: torch.Generator) -> Tensor:
    mask = torch.rand(shape, generator=generator) < mask_probability
    row_has_mask = mask.any(dim=1)
    if bool(row_has_mask.all()):
        return mask
    missing_rows = torch.where(~row_has_mask)[0]
    columns = torch.randint(0, shape[1], (len(missing_rows),), generator=generator)
    mask[missing_rows, columns] = True
    return mask


def _batch_iterator(
    features: Tensor,
    targets: Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> list[tuple[Tensor, Tensor]]:
    permutation = torch.randperm(targets.shape[0], generator=generator)
    return [
        (
            features[permutation[start : start + batch_size]],
            targets[permutation[start : start + batch_size]],
        )
        for start in range(0, targets.shape[0], batch_size)
    ]


def _masked_prediction(
    model: MaturityGraphAutoencoder,
    values: Tensor,
    target_shape: torch.Size,
    mask_probability: float,
    random_seed: int,
) -> MaskedPrediction:
    generator = torch.Generator().manual_seed(random_seed)
    mask = _random_mask(target_shape, mask_probability, generator)
    reconstructed, _ = model(_model_input(values, mask))
    return MaskedPrediction(reconstructed.numpy(), mask.numpy().astype(np.bool_))


def _date_ordered_split(
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
    frame = pd.DataFrame(encoded, columns=[f"GE{i}" for i in range(1, encoded.shape[1] + 1)])
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
    frame["split"] = pd.to_datetime(frame["date"]).map(
        dict(zip(pd.to_datetime(panel.index), split_labels, strict=True))
    )
    if mask is not None:
        mask_frame = pd.DataFrame(mask, index=panel.index, columns=panel.columns)
        mask_long = mask_frame.stack().rename("is_masked").reset_index()
        mask_long = mask_long.rename(
            columns={mask_long.columns[0]: "date", mask_long.columns[1]: "maturity_years"}
        )
        frame = frame.merge(mask_long, on=["date", "maturity_years"], how="inner")
        frame = frame.loc[frame["is_masked"]].drop(columns=["is_masked"])
    return frame.loc[:, ["date", "country", "maturity_years", "yield", "fitted_yield", "split"]]


def _metrics_frame(
    reconstruction: pd.DataFrame,
    country: str,
    latent_dim: int,
    hidden_dim: int,
    n_layers: int,
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
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
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


def _validate_hyperparameters(
    latent_dim: int,
    hidden_dim: int,
    n_layers: int,
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
    if latent_dim <= 0 or hidden_dim <= 0 or n_layers <= 0:
        raise ValueError("GNN dimensions must be positive")
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
