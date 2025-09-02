#!/bin/bash

# EWCL API Deployment Script for Fly.io
set -e

echo "🚀 Deploying EWCL API to Fly.io..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl not found. Please install it first:"
    echo "   curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "🔐 Please log in to Fly.io first:"
    echo "   flyctl auth login"
    exit 1
fi

# Build and deploy
echo "📦 Building Docker image and deploying..."
flyctl deploy --dockerfile ./Dockerfile

echo "✅ Deployment complete!"
echo "🌐 Your API should be available at: https://ewcl-api.fly.dev"
echo ""
echo "🔍 Test endpoints:"
echo "   Health: https://ewcl-api.fly.dev/healthz"
echo "   Docs: https://ewcl-api.fly.dev/docs"
echo ""
echo "📊 Model endpoints:"
echo "   EWCLv1: https://ewcl-api.fly.dev/ewcl/analyze-fasta/ewclv1"
echo "   EWCLv1-M: https://ewcl-api.fly.dev/ewcl/analyze-fasta/ewclv1m"
echo "   EWCLv1-P3: https://ewcl-api.fly.dev/ewcl/analyze-pdb/ewclv1p3"
echo "   ClinVar: https://ewcl-api.fly.dev/clinvar/analyze"
