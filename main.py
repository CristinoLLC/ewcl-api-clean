from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from models.ewcl_physics import compute_ewcl_from_pdb
from models.qwip3d import run_qwip_on_pdb, compute_qwip3d
from utils.qwip3d_disorder import predict_disorder
import pandas as pd
import json
import os
import tempfile
import io
import shutil
from typing import List
from Bio.PDB import PDBParser
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ────────── START
