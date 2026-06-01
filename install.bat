@echo off
echo Installing Teachable Machine Pro Dependencies...
echo ================================================

REM Backend
echo.
echo Installing Backend Requirements...
cd backend
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
call deactivate
cd ..

REM Frontend
echo.
echo Installing Frontend Requirements...
cd frontend
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
call deactivate
cd ..

echo.
echo Installation Complete!
echo.
echo To run:
echo   Backend:  cd backend ^&^& venv\Scripts\activate ^&^& uvicorn main:app --reload
echo   Frontend: cd frontend ^&^& venv\Scripts\activate ^&^& streamlit run app.py
pause