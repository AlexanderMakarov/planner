from flask import Flask
app = Flask('planner')

app.config['JSON_SORT_KEYS'] = False

import predict_view
