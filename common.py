"""Shared data loading and feature definitions."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "enterprise_full_dataset.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "Productivity_Class"
CLASS_CODE = {"Low": 0, "Medium": 1, "High": 2}
EQUITY = ["State_Equity_Ratio", "Private_Equity_Ratio",
          "Institutional_Equity_Ratio", "Foreign_Equity_Ratio"]
REFERENCE_SHARE = "Foreign_Equity_Ratio"  # omitted from model inputs

DIMENSIONS = {
    "Innovation": ["Patent_Count", "R&D_Expenditure", "Innovation_Index", "New_Product_Ratio"],
    "Digital": ["Digital_Investment", "AI_Adoption_Level", "Automation_Level", "Digital_Platform_Usage"],
    "Green": ["Energy_Efficiency", "Carbon_Emission", "ESG_Score", "Waste_Management_Score"],
    "Integration": ["SupplyChain_Integration", "Industry_Collaboration", "Cross_Industry_Integration"],
    "Ownership": ["State_Equity_Ratio", "Private_Equity_Ratio", "Institutional_Equity_Ratio", "ERR"],
    "Risk": ["Risk_Score", "Debt_Ratio"],
    "Operational": ["Revenue_per_Employee", "Asset_Utilization", "Cost_Efficiency"],
}

# Recovered labelling rule, Eq. (2)
RULE = [(0.25, "Innovation_Index"), (0.20, "AI_Adoption_Level"),
        (0.15, "Energy_Efficiency"), (0.15, "Asset_Utilization"),
        (0.10, "Digital_Platform_Usage"), (0.10, "SupplyChain_Integration"),
        (-0.05, "Risk_Score")]
CUTS = (0.50, 0.70)

SEEDS = (42, 43, 44)


def load(path=DATA):
    d = pd.read_csv(path)
    assert d.shape == (5275, 29), f"unexpected shape {d.shape}"
    return d


def indicators(d):
    """All 25 numeric indicators."""
    return [c for c in d.columns if c not in ("Firm_ID", "Year", "Industry", TARGET)]


def modelled(d):
    """24 modelled indicators (reference equity share dropped)."""
    return [c for c in indicators(d) if c != REFERENCE_SHARE]


def target(d):
    return d[TARGET].map(CLASS_CODE).to_numpy()
