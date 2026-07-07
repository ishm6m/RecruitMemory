# Dockerfile — the recipe for building a self-contained image of the app.
# Anyone with Docker can build this and run RecruitMemory without installing
# Python, pip, or any of our dependencies by hand.

# Start from a slim official Python image (matches our local Python 3.10).
FROM python:3.10-slim

# All app files will live in /app inside the container.
WORKDIR /app

# Install dependencies FIRST (as its own layer). Docker caches this step, so
# rebuilds are fast as long as requirements.txt hasn't changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source code in.
COPY . .

# The app listens on port 8000 inside the container.
EXPOSE 8000

# Start the web server. 0.0.0.0 means "accept connections from outside the
# container" (127.0.0.1 would only be reachable from inside it).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
