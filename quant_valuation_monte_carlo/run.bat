@echo off
echo ============================================================
echo   QuantValuation & Monte Carlo Brownian Motion Dashboard
echo ============================================================
echo Installation des dependances requises...
pip install -r requirements.txt
echo.
echo Lancement du tableau de bord Streamlit...
streamlit run app.py
pause
