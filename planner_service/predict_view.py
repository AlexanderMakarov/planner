from planner_service import app
from flask import request, jsonify


@app.route('/', methods=['GET'])
def index():
    s = request.args.get('s')
    if not s or len(s) == 0:
        return jsonify({"error": "Can't parse 's' parameter from URL."})
    series = [int(x) for x in request.args.get('s')]
    trend = sum(series) / len(series)
    return jsonify({
        "input": series,
        "trend": trend,
        "prediction": series[-1] + trend
    })
