FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# ===== 安装系统依赖 =====
RUN apt-get update && apt-get install -y \
    git \
    python3-pyqt5 \
    x11-utils \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgtk2.0-dev libusb-1.0-0-dev udev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ===== 升级 pip =====
RUN pip install --upgrade pip

# ===== 安装 Python 包 =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
