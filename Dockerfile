FROM python:3.10-slim

WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Default port for Render
ENV PORT=10000
EXPOSE 10000

# Start the dashboard
CMD ["python", "app.py"]
