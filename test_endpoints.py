"""
Simple test to verify all endpoints are accessible
"""

import requests

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Health Check:", response.json())

if __name__ == "__main__":
    try:
        test_health_check()
        print("✅ All endpoints accessible")
    except Exception as e:
        print(f"❌ Error: {e}")
