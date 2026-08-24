#!/bin/bash
echo "============================================================"
echo "  QuantValuation & Monte Carlo Brownian Motion Dashboard"
echo "============================================================"
echo "Installation des dépendances requises..."
pip install -r requirements.txt
echo ""
echo "Lancement du tableau de bord Streamlit..."
streamlit run app.py
