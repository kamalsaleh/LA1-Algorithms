FROM python:3-alpine3.15

# Set the working directory in the container
WORKDIR /app

# Copy all the files located in the current directory to the container working directory
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5001

# Run the application
CMD ["python", "./app.py"]

# Build the Docker file on a local machine
# docker build -t kamalsaleh/linear-algebra-ws24:latest .

# Run the Docker container after building the Docker file
# docker container run -p 5001:5001 kamalsaleh/linear-algebra-ws24:latest
