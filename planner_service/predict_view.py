from __init__ import app
from flask import request, jsonify
import logging
import prophet


@app.route('/', methods=['GET'])
def index():
    s = request.args.get('s')
    if not s or len(s) == 0:
        return jsonify({"error": "Can't parse 's' parameter from URL."})
    series = [int(x) for x in request.args.get('s')]
    logging.info(f"Called with {series}")
    mean = sum(series) / len(series)
    if len(series) < 2:
        trend = 0
        is_trend_crossed_mean = False
    else:
        trend = series[-1] - series[-2]
        is_trend_crossed_mean = (series[-1] - mean) * (series[-2] - mean) < 0
    return jsonify({
        "input": series,
        "mean": mean,
        "trend": trend,
        "prediction": mean if is_trend_crossed_mean else series[-1] + trend 
    })
