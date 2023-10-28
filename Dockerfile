# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV MODEL_PATH=/app/random_forest_model.pkl

# Run app.py when the container launches
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]


# Build & Run commands
# docker build -t winequality_prediction_app .
# docker run -p 8080:8080 winequality_prediction_app

# In the cloud - It can be pulled from my Docker Hub
# docker pull dawei7/winequality_prediction_app

# https://wine-predict.azurewebsites.net/