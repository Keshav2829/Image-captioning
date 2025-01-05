# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./core/model /app/core/model
COPY ./data/tokenizer /app/data/tokenizer
COPY ./data/weights/decoder /app/data/weights/decoder
COPY ./data/weights/encoder /app/data/weights/encoder
COPY .streamlit /app/.streamlit
COPY ./infer_config.yaml /app/infer_config.yaml
COPY ./requirements.txt /app/requirements.txt
COPY ./infer.py /app/infer.py
COPY ./streamlit_infer.py /app/streamlit_infer.py

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 7860
# Define environment variable

# Run app.py when the container launches
CMD ["streamlit", "run", "streamlit_infer.py"]
