@echo off
echo ============================================
echo  智能图片批处理流水线工具
echo  Python + Pillow + OpenCV + Streamlit
echo ============================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo [1/3] 创建虚拟环境...
    python -m venv venv
)

echo [2/3] 激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo [3/3] 启动 Streamlit 应用...
streamlit run app.py --server.port 8501 --server.maxUploadSize 4096

pause
