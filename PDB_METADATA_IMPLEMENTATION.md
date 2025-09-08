# PDB Metadata Enrichment Implementation

## Overview

This implementation adds PDB metadata enrichment to the EWCLv1-P3 PDB analysis endpoint without affecting model performance or inference speed. The metadata is fetched concurrently and with short timeouts to ensure it never blocks the core prediction functionality.

## Key Features

### ✅ **Safe & Non-Blocking**
- Metadata fetch runs **concurrently** with model inference
- **Short timeout** (1.5s default) prevents blocking
- **Best-effort** approach - returns predictions even if metadata fails
- **Cached** results to avoid repeated API calls

### ✅ **Comprehensive Metadata**
- **RCSB API integration** for experimental structures
- **Local analysis** for predicted models (AlphaFold)
- **pLDDT statistics** for AlphaFold structures
- **B-factor statistics** for X-ray structures
- **Resolution, R-factors, method** from RCSB

### ✅ **Smart PDB ID Detection**
- Extracts PDB ID from filename (e.g., `1CRN.pdb` → `1crn`)
- Parses PDB header for ID if filename doesn't match
- Handles various file naming conventions

## Implementation Details

### Core Functions

#### `_maybe_guess_pdb_id(filename, data)`
```python
# Extracts PDB ID from filename or PDB header
pdb_id = _maybe_guess_pdb_id("1CRN.pdb", pdb_bytes)
# Returns: "1crn" or None
```

#### `_fetch_metadata(pdb_id)`
```python
# Fetches metadata from RCSB API with timeout
meta = await _fetch_metadata("1crn")
# Returns: {"method": "X-RAY DIFFRACTION", "resolution_angstrom": 1.5, ...}
```

#### `_extract_local_metadata(pdb_data, residues)`
```python
# Extracts local statistics from pLDDT/B-factor values
meta = _extract_local_metadata(af_data, residues)
# Returns: {"plddt_mean": 87.2, "plddt_std": 5.1, ...}
```

### Response Structure

The enhanced endpoint now returns:

```json
{
  "id": "1CRN.pdb",
  "model": "ewclv1p3",
  "source": "xray",
  "metric_name": "bfactor",
  "length": 46,
  "residues": [...],
  "diagnostics": {
    "pdb_id": "1crn",
    "note": "scores computed; metadata best-effort",
    "feature_count": 302,
    "parser_version": "fresh_complete_implementation",
    
    // RCSB metadata (if available)
    "method": "X-RAY DIFFRACTION",
    "resolution_angstrom": 1.5,
    "r_work": 0.18,
    "r_free": 0.22,
    "source": "rcsb",
    
    // OR local metadata (if no RCSB data)
    "method": "Predicted model",
    "plddt_mean": 87.2,
    "plddt_median": 88.1,
    "plddt_min": 78.5,
    "plddt_max": 95.3,
    "plddt_std": 5.1,
    "source": "local_analysis"
  }
}
```

## Configuration

### Environment Variables

```bash
# Metadata fetch timeout (seconds)
PDB_META_TIMEOUT_SEC=1.5

# Disable metadata fetching entirely
DISABLE_PDB_METADATA=true
```

### Dependencies

Added to `requirements-backend.txt`:
```
httpx==0.27.0  # For async HTTP requests to RCSB API
```

## Usage Examples

### Basic Usage
```python
import httpx

# Upload PDB file
files = {"file": open("1CRN.pdb", "rb")}
response = httpx.post("http://localhost:8000/ewcl/analyze-pdb/ewclv1-p3", files=files)
result = response.json()

# Access metadata
diagnostics = result["diagnostics"]
if "resolution_angstrom" in diagnostics:
    print(f"Resolution: {diagnostics['resolution_angstrom']} Å")
if "plddt_mean" in diagnostics:
    print(f"Mean pLDDT: {diagnostics['plddt_mean']}")
```

### Frontend Integration
```javascript
const analyzePDB = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/ewcl/analyze-pdb/ewclv1-p3', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  // Display metadata in UI
  const metadata = result.diagnostics;
  if (metadata.resolution_angstrom) {
    showResolution(metadata.resolution_angstrom);
  }
  if (metadata.plddt_mean) {
    showConfidence(metadata.plddt_mean);
  }
  
  return result;
};
```

## Performance Characteristics

### ✅ **Zero Impact on Core Performance**
- Model inference runs **unchanged**
- Feature extraction runs **unchanged**
- Metadata fetch is **completely parallel**

### ✅ **Fast Response Times**
- **1.5s timeout** ensures quick responses
- **Cached results** for repeated requests
- **Graceful degradation** if RCSB is slow

### ✅ **Resource Efficient**
- **Single HTTP client** per request
- **Concurrent requests** to RCSB API
- **Memory efficient** with small metadata objects

## Error Handling

### Network Issues
- **Timeout**: Returns predictions without metadata
- **Connection error**: Continues with local metadata
- **Invalid PDB ID**: Falls back to local analysis

### Data Issues
- **Malformed PDB**: Still returns predictions
- **Missing fields**: Graceful handling with defaults
- **API changes**: Robust parsing with fallbacks

## Testing

Run the test script to verify functionality:

```bash
python test_pdb_metadata.py
```

This tests:
- PDB ID extraction from filenames and headers
- Local metadata extraction for AlphaFold/X-ray
- External metadata fetching from RCSB
- Error handling and timeouts

## Future Enhancements

### Potential Improvements
1. **Local PDB parsing**: Extract resolution from mmCIF headers
2. **Caching layer**: Redis cache for frequently accessed PDBs
3. **Background processing**: Fire-and-forget metadata updates
4. **Additional sources**: UniProt, PDBj, or other databases

### Configuration Options
```python
# Future configuration options
PDB_METADATA_ENABLED=true
PDB_METADATA_CACHE_TTL=3600
PDB_METADATA_SOURCES=rcsb,uniprot
PDB_METADATA_TIMEOUT=2.0
```

## Scientific Value

This implementation provides valuable context for protein structure analysis:

- **Quality assessment**: Resolution and R-factors indicate structure quality
- **Method validation**: X-ray vs NMR vs predicted structures
- **Confidence metrics**: pLDDT statistics for AlphaFold models
- **Research context**: Links to original experimental data

The metadata enriches the scientific interpretation without compromising the core disorder prediction functionality.
