# Step 1: Base image (Python 3.11 slim version)
FROM python:3.11-slim

# Step 2: Working directory set করছ
WORKDIR /app

# Step 3: requirements.txt copy করছ (source destination দটই দত হব)
COPY requirements.txt .

# Step 4: Python packages install করছ
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: সব app files copy করছ
COPY . .

# Step 6: Port expose করছ
EXPOSE 8000

# Step 7: App run করর command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
