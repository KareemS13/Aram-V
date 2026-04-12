"""
Flask dashboard for Armenia CPI Inflation Forecasting Pipeline.

Run:  python app.py
Then open http://localhost:5000
"""

import os
import subprocess
import threading

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(BASE_DIR, "outputs", "figures")
TABLE_DIR = os.path.join(BASE_DIR, "outputs", "tables")

# Shared pipeline state (single-user local tool)
_job = {"running": False, "log": [], "done": False, "error": None}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Static figure serving
# ---------------------------------------------------------------------------

@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory(FIG_DIR, filename)


# ---------------------------------------------------------------------------
# Data API
# ---------------------------------------------------------------------------

def _read_csv(name):
    path = os.path.join(TABLE_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@app.route("/api/history")
def api_history():
    df = _read_csv("cpi_history.csv")
    if df is None:
        return jsonify({"error": "Run the pipeline first."}), 404
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/forecast")
def api_forecast():
    df = _read_csv("forecast_point_ci.csv")
    if df is None:
        return jsonify({"error": "Run the pipeline first."}), 404
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/drivers")
def api_drivers():
    df = _read_csv("shap_driver_summary.csv")
    if df is None:
        return jsonify({"error": "Run the pipeline first."}), 404
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/metrics")
def api_metrics():
    df = _read_csv("model_cv_metrics.csv")
    if df is None:
        return jsonify({"error": "Run the pipeline first."}), 404
    return jsonify(df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    return jsonify(_job)


@app.route("/api/run", methods=["POST"])
def api_run():
    if _job["running"]:
        return jsonify({"error": "Pipeline already running."}), 409

    opts = request.get_json(silent=True) or {}
    no_tune = opts.get("no_tune", False)
    no_cv   = opts.get("no_cv", False)

    def _run():
        _job["running"] = True
        _job["log"]     = []
        _job["done"]    = False
        _job["error"]   = None

        cmd = ["python", "inflation.py"]
        if no_tune:
            cmd.append("--no-tune")
        if no_cv:
            cmd.append("--no-cv")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=BASE_DIR,
            )
            for line in proc.stdout:
                _job["log"].append(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                _job["error"] = "Pipeline exited with errors — check log."
        except Exception as exc:
            _job["error"] = str(exc)
        finally:
            _job["running"] = False
            _job["done"]    = True

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"message": "Pipeline started."})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
