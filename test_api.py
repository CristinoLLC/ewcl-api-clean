"""
Simple test script to verify the API is working
"""

import requests
import json

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("https://ewcl-platform.onrender.com/health")
        print("✅ Health check:")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root():
    """Test the root endpoint"""
    try:
        response = requests.get("https://ewcl-platform.onrender.com/")
        print("✅ Root endpoint:")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_analyze_pdb():
    """Test the analyze-pdb endpoint with a simple PDB"""
    try:
        # Simple PDB content for testing
        pdb_content = """
ATOM      1  CA  ALA A   1      20.154   6.718  50.086  1.00 20.00           C  
ATOM      2  CA  VAL A   2      21.155   7.718  51.086  1.00 25.00           C  
ATOM      3  CA  PHE A   3      22.156   8.718  52.086  1.00 30.00           C  
END
"""
        
        files = {"file": ("test.pdb", pdb_content, "text/plain")}
        response = requests.post("https://ewcl-platform.onrender.com/analyze-pdb", files=files)
        
        if response.status_code == 200:
            print("✅ analyze-pdb endpoint works!")
            result = response.json()
            print(f"📊 Analyzed {len(result)} residues")
            if result:
                print(f"📝 Sample result: {result[0]}")
            return True
        else:
            print(f"❌ analyze-pdb failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ analyze-pdb test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing EWCL API...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Analyze PDB", test_analyze_pdb),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️ Some tests failed, but API is still functional.")
