"""
Local development runner for EWCL API
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting EWCL API locally...")
    print("📡 Available at: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/health")
    print("📚 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
