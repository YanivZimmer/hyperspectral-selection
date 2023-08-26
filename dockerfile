FROM python:3.11
RUN python --version
RUN mkdir /usr/bin/code
COPY . /usr/bin/code
#ENV PYTHONPATH "${PYTHONPATH}:/usr/bin/code"
#RUN pip install --no-cache-dir --upgrade pip
RUN ls
#RUN virtualenv /ve
#RUN /ve/bin/pip install -r /usr/bin/code/requirements.txt
#CMD ["/ve/bin/python", "/usr/bin/code/main_multiple.py"]
RUN pip install -r /usr/bin/code/requirements.txt
#RUN export PYTHONPATH=/usr/bin/code
WORKDIR /usr/bin/code
CMD ./startup.sh
