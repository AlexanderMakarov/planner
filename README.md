## Setup

Run `setup.py`.

## Run

Cd to "predict_planner" and execute `run.py`.

Execute in browser something like "http://127.0.0.1:5000/?s=1231".


## Run from Docker

$ cd planner_service
$ docker run -e PORT=8080 -t -p 8080:8080 planner_service