# Vercel serverless function entry point for eBay Seller Tools
import sys
import os

# Add the parent directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from app import app

# Vercel expects this specific function signature
def handler(request, context):
    """Vercel serverless function handler"""
    return app(request.environ, lambda *args: None)

# For local testing
if __name__ == "__main__":
    app.run(debug=True, port=5000)
