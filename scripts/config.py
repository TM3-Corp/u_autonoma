"""
Canvas API Configuration for Universidad Autónoma de Chile
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Canvas API Configuration - TEST Environment (Updated Dec 2025)
# Credentials loaded from .env file (never commit secrets!)
API_URL = os.getenv('CANVAS_API_URL', 'https://uautonoma.test.instructure.com')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')

if not API_TOKEN:
    raise ValueError("CANVAS_API_TOKEN not found! Copy .env.example to .env and add your token.")

# Account hierarchy
ACCOUNT_ID_UNIVERSIDAD = 1
ACCOUNT_ID_PREGRADO = 46
ACCOUNT_ID_SEDE_PROVIDENCIA = 176
ACCOUNT_ID_CARRERA = 719  # Ing. en Control de Gestión

# High-potential course IDs (resources >= 50 AND students > 0)
HIGH_POTENTIAL_COURSES = [
    86689,  # GESTIÓN DEL TALENTO-P01
    86161,  # INGLÉS II-P01
    86153,  # PLANIFICACIÓN ESTRATÉGICA-P02
    86155,  # DERECHO TRIBUTARIO-P01
    86177,  # PLANIFICACIÓN ESTRATÉGICA-P01
    86179,  # DERECHO TRIBUTARIO-P02
    76755,  # PENSAMIENTO MATEMÁTICO-P03
    86676,  # FUND DE BUSINESS ANALYTICS-P01
    86677,  # MATEMÁTICAS PARA LOS NEGOCIOS-P01
    86686,  # MATEMÁTICAS PARA LOS NEGOCIOS-P03
    85822,  # LAB DE CONTABILIDAD Y COSTOS-P01
    86673,  # LAB DE CONTABILIDAD Y COSTOS-P04
    86005,  # TALL DE COMPETENCIAS DIGITALES-P01
    86020,  # TALL DE COMPETENCIAS DIGITALES-P02
    86670,  # FUND DE BUSINESS ANALYTICS-P02
    85825,  # GESTIÓN DEL TALENTO-P02
    86675,  # INGLÉS II-P03
    85481,  # PENSAMIENTO MATEMÁTICO-P05
    82725,  # LAB DE METOD CONT EV. CICLO-P03
    84947,  # EST APLIC A BUSINESS ANALYTICS-P04
    84939,  # EST APLIC A BUSINESS ANALYTICS-P01
]

# Data paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
