# Base image: slim Python 3.11 keeps the dashboard image small.
FROM python:3.11-slim

# All code and installs live under /app inside the container.
WORKDIR /app

# Install dependencies first, as their own layer, so Docker caches them and
# doesn't reinstall on every code change (only when requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (data/ and rdf_output/ are excluded via .dockerignore).
COPY . .

# Streamlit's default port.
EXPOSE 8501

# Launch the dashboard, bound to 0.0.0.0 so it's reachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]