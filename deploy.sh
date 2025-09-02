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
echo "✓ EWCL API deployed."
echo "  Test endpoints:"
echo "   Health:   https://ewcl-api-clean.fly.dev/healthz"
echo "   Models:   https://ewcl-api-clean.fly.dev/models"
echo "   EWCLv1:   https://ewcl-api-clean.fly.dev/ewcl/analyze-fasta/ewclv1"
echo "   EWCLv1-M: https://ewcl-api-clean.fly.dev/ewcl/analyze-fasta/ewclv1-M"
echo "   EWCLv1-P3: https://ewcl-api-clean.fly.dev/ewcl/analyze-pdb/ewclv1p3"
echo "   EWCLv1-C: https://ewcl-api-clean.fly.dev/clinvar/analyze-variants/ewclv1-C"
echo ""
