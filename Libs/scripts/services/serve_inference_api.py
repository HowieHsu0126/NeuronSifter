from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.models.networks.shared_latent_dual_encoder import SharedLatentDualEncoder
from Libs.utils.data.raw_data_registry import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the exported NeuronSifter production model with a lightweight JSON API.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Optional override for the exported inference bundle directory.",
    )
    return parser.parse_args()


def load_bundle(config: dict, bundle_dir: Path) -> dict[str, object]:
    serving_config = config["serving_registry"]
    filenames = serving_config["bundle_filenames"]
    preprocessing_manifest = json.loads((bundle_dir / filenames["preprocessing_manifest"]).read_text(encoding="utf-8"))
    label_dictionary = json.loads((bundle_dir / filenames["label_dictionary"]).read_text(encoding="utf-8"))
    model_card = json.loads((bundle_dir / filenames["model_card"]).read_text(encoding="utf-8"))
    entity_dictionary = pd.read_parquet(bundle_dir / filenames["entity_dictionary"])
    checkpoint = torch.load(bundle_dir / filenames["model_weights"], map_location="cpu")
    model = SharedLatentDualEncoder(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    lookup_by_modality: dict[str, dict[str, int]] = {}
    for modality in serving_config["supported_modalities"]:
        current_entity_frame = entity_dictionary.loc[entity_dictionary["modality"] == modality].copy()
        lookup_by_modality[modality] = {
            str(entity_id): index
            for index, entity_id in enumerate(current_entity_frame["canonical_entity_id"].astype(str).tolist())
        }
    return {
        "preprocessing_manifest": preprocessing_manifest,
        "label_dictionary": label_dictionary,
        "model_card": model_card,
        "entity_dictionary": entity_dictionary,
        "lookup_by_modality": lookup_by_modality,
        "model": model,
    }


def predict_one(sample_payload: dict[str, object], serving_config: dict, bundle: dict[str, object]) -> dict[str, object]:
    for field_name in ["modality", "feature_namespace", "feature_values"]:
        if field_name not in sample_payload:
            raise ValueError(f"Request is missing required field: {field_name}")
    modality = str(sample_payload["modality"])
    feature_namespace = str(sample_payload["feature_namespace"])
    feature_values = sample_payload["feature_values"]
    if modality not in serving_config["supported_modalities"]:
        raise ValueError(f"Unsupported modality: {modality}")
    expected_namespace = serving_config["request_feature_namespace"][modality]
    if feature_namespace != expected_namespace:
        raise ValueError(f"Unexpected feature_namespace for {modality}: {feature_namespace}")
    if not isinstance(feature_values, dict):
        raise ValueError("feature_values must be an object mapping canonical feature ids to numeric values.")
    lookup = bundle["lookup_by_modality"][modality]
    unknown_features = sorted(str(feature_id) for feature_id in feature_values if str(feature_id) not in lookup)
    if unknown_features:
        raise ValueError(f"Unknown features for {modality}: {unknown_features[:10]}")
    if len(feature_values) < int(serving_config["minimum_recognized_feature_count"][modality]):
        raise ValueError(f"Insufficient recognized features for {modality}.")
    vector = torch.zeros((1, len(lookup)), dtype=torch.float32)
    for feature_id, value in feature_values.items():
        vector[0, lookup[str(feature_id)]] = float(value)
    with torch.no_grad():
        logits, latent = bundle["model"](modality, vector)
        probability = float(torch.sigmoid(logits)[0].cpu().item())
        predicted_label = bundle["label_dictionary"]["1"] if probability >= 0.5 else bundle["label_dictionary"]["0"]
    return {
        "predicted_label": predicted_label,
        "ad_probability": probability,
        "latent_embedding": latent[0].cpu().tolist(),
        "model_version": bundle["preprocessing_manifest"]["model_version"],
        "preprocessing_version": bundle["preprocessing_manifest"]["preprocessing_version"],
    }


def app_factory(config: dict, bundle: dict[str, object]):
    serving_config = config["serving_registry"]

    class RequestHandler(BaseHTTPRequestHandler):
        def _json_response(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Request body is empty.")
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json_response(HTTPStatus.OK, {"status": "ok"})
                return
            if self.path == "/model_card":
                self._json_response(HTTPStatus.OK, bundle["model_card"])
                return
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "unknown_endpoint"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json_body()
                if self.path == "/predict":
                    self._json_response(HTTPStatus.OK, predict_one(payload, serving_config, bundle))
                    return
                if self.path == "/predict_batch":
                    samples = payload.get("samples")
                    if not isinstance(samples, list):
                        raise ValueError("predict_batch requires a samples list.")
                    predictions = [predict_one(sample_payload, serving_config, bundle) for sample_payload in samples]
                    self._json_response(HTTPStatus.OK, {"predictions": predictions})
                    return
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "unknown_endpoint"})
            except Exception as exc:  # noqa: BLE001
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format_string: str, *args: object) -> None:
            return

    return RequestHandler


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    serving_config = config["serving_registry"]
    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else (
        REPO_ROOT
        / serving_config["production_root_template"].format(
            benchmark_name=config["benchmark_registry"]["benchmark_name"],
            model_version=serving_config["default_model_version"],
        )
    ).resolve()
    bundle = load_bundle(config, bundle_dir)
    server = ThreadingHTTPServer(
        (serving_config["server"]["host"], int(serving_config["server"]["port"])),
        app_factory(config, bundle),
    )
    pipeline_logger.artifact("serving_bundle_loaded", artifact=bundle_dir)
    pipeline_logger.progress(
        "server_listening",
        host=serving_config["server"]["host"],
        port=int(serving_config["server"]["port"]),
    )
    print(f"serving_bundle={bundle_dir}")
    print(f"listening=http://{serving_config['server']['host']}:{serving_config['server']['port']}")
    server.serve_forever()

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
