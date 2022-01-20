
# https://cloud.google.com/run/docs/quickstarts/build-and-deploy/python
# NOTE "slim" images doesn't work for pystan - compilation erorrs.
FROM python:3.9

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# RUN pip install cython plotly numpy xlrd pandas requests pytz uuid matplotlib tqdm scikit-learn seaborn scipy \
#    && pip install pymeeus ujson korean-lunar-calendar hijri-converter ephem convertdate setuptools-git pystan==2.19.1.1 LunarCalendar holidays cmdstanpy \
#    && pip install prophet

# https://discourse.mc-stan.org/t/pystan-3-gcc-compilers-and-docker-images/21342/2
RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install -y gcc
RUN rm -rf /var/lib/apt/lists/*

# Install production dependencies.
# Don't use requirements.txt but first pre-install 'pystan' dependencies - it is too capricious.
RUN pip install --no-cache-dir flask gunicorn
RUN pip install --no-cache-dir cython plotly numpy pandas matplotlib tqdm scipy LunarCalendar holidays convertdate \
    setuptools setuptools-git python-dateutil wheel cmdstanpy
RUN pip install --no-cache-dir pystan==2.19.1.1 prophet

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 run:app