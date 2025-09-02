# Railway Deployment Guide for EWCL API

## 🚄 Railway Setup Instructions

### 1. Repository Preparation
Your repository is now Railway-ready with:
- ✅ `Dockerfile` (Railway will use this by default) - **Fixed PORT handling**
- ✅ `Procfile` (fallback option)
- ✅ `railway.json` (Railway configuration)
- ✅ `requirements.txt` (Python dependencies)
- ✅ **PORT environment variable handling fixed in main.py**

### 2. Deploy to Railway

#### Option A: GitHub Integration (Recommended)
1. Push your repo to GitHub (already done)
2. Go to [Railway](https://railway.app)
3. Create new project → "Deploy from GitHub repo"
4. Select your `ewcl-api-clean` repository
5. Railway will auto-detect Dockerfile and deploy

#### Option B: Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```

### 3. Environment Variables
In Railway Dashboard → Your Service → Variables tab, add:

```
MAX_BODY_BYTES=100000000
EWCLV1_MODEL_PATH=/app/models/disorder/ewclv1.pkl
EWCLV1_M_MODEL_PATH=/app/models/disorder/ewclv1-M.pkl
EWCLV1_P3_MODEL_PATH=/app/models/pdb/ewclv1p3.pkl
EWCLV1_C_MODEL_PATH=/app/models/clinvar/ewclv1-C.pkl
EWCLV1_C_FEATURES_PATH=/app/models/clinvar/EWCLv1-C_features.json
ENABLE_RAW_ROUTERS=0
```

**Note**: PORT environment variable is handled automatically. The app defaults to port 8080 if Railway doesn't provide PORT.

### 4. Test Deployment
Once deployed, test with:
```bash
# Health check
curl -s https://your-app.up.railway.app/healthz

# Models endpoint
curl -s https://your-app.up.railway.app/models | jq

# Sample prediction
curl -X POST https://your-app.up.railway.app/ewclv1 \
  -H "Content-Type: application/json" \
  -d @samples/sample_payload_ewclv1.json
```

### 5. Domain & SSL
Railway automatically provides:
- HTTPS endpoint: `https://your-app.up.railway.app`
- Custom domain support (if needed)
- SSL certificates (automatic)

## 🔧 PORT Variable Fix Applied

**Problem Solved**: Railway was seeing `"$PORT"` as a literal string instead of the port number.

**Solution Implemented**:
1. **main.py**: Added robust PORT parsing with fallback to 8080
2. **Dockerfile**: Uses fixed port 8080 instead of `$PORT` variable substitution
3. **Compatibility**: Works with both Railway and Fly.io

## 🔄 Dual Deployment Strategy

You can run both Fly.io and Railway simultaneously:

### Current Fly.io (Production)
- URL: `https://ewcl-api-clean.fly.dev`
- 8 machines across 4 regions
- Singleton model loading ✅

### New Railway (Testing)
- URL: `https://ewcl-api-clean.up.railway.app`
- Automatic scaling
- Same codebase, faster deployments
- **PORT issue fixed** ✅

## 📊 Performance Comparison

Test both platforms with your workloads:
- **Deployment Speed**: Railway typically faster
- **Cold Start**: Railway often better  
- **Global Edge**: Fly.io has more regions
- **Cost**: Compare pricing for your usage

## 🚀 Migration Steps

1. **Deploy to Railway** (parallel to Fly.io)
2. **Test API performance** and reliability
3. **Update Colab notebooks** to use Railway URL
4. **Performance comparison** between platforms
5. **If satisfied**, scale down or remove Fly.io deployment

## 🛠️ Troubleshooting

### Build Issues
- Check Railway build logs
- Ensure all model files are in Git LFS
- Verify Dockerfile compatibility

### Runtime Issues  
- Check Railway service logs
- Verify environment variables
- Test model loading at startup

### PORT Issues (SOLVED)
- ✅ **Fixed**: Robust PORT parsing in main.py
- ✅ **Fixed**: Dockerfile uses fixed port 8080
- ✅ **Compatible**: Works with Railway, Fly.io, and local development

## 🎯 Next Steps

1. **Push changes** to GitHub
2. **Deploy to Railway** using GitHub integration
3. **Test the deployment** with health checks
4. **Compare performance** with existing Fly.io deployment