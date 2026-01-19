# Dockerfile for Flight Prediction Project
FROM python:3.11-slim
# Set the working directory
WORKDIR /app
# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
# Copy the entire project directory into the container
COPY flask_app/ ./flask_app/
COPY src/ ./src/
# Expose the port that the Flask app will run on
EXPOSE 5000
# Command to run the Flask app  
CMD ["python3", "flask_app/app.py"]