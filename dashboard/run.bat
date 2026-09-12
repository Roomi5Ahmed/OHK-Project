@echo off
echo Installing dashboard dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit dashboard...
python -m streamlit run app.py
pause
