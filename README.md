# 华矩EL组件裁剪工具

基于 OpenCV 的太阳能电池板 EL（电致发光）图像自动裁剪工具。本地 Flask Web 应用，支持批量透视校正裁剪。

## 功能特点

- **自动检测面板边界**：v4.0 检测内核，Hough 直线检测为主，亮度阈值法为备选
- **透视校正裁剪**：四角定位 + 透视变换，输出矫正后的矩形图像
- **手动编辑**：Canvas 拖拽四角控制点微调裁剪区域
- **批量处理**：多文件拖放上传，多线程并行处理
- **自学习系统**：记录手动修正并自动优化后续检测
- **GPU 加速**：自动检测 OpenCL 并启用加速

## 环境要求

- Python 3.8+
- NumPy
- OpenCV (`opencv-python` 或 `opencv-contrib-python`)
- Flask

## 安装

```bash
pip install numpy opencv-python flask
```

## 运行

```bash
python app.py
```

启动后自动打开浏览器访问 `http://localhost:5000`。

## 使用方法

1. 拖放或选择 EL 图像文件上传
2. 工具自动检测面板边框并裁剪
3. 如需调整，点击图像进入编辑模式，拖动四角控制点
4. 按 **D** 键或点击按钮应用裁剪
5. 批量处理时可一键导出所有结果

## 项目结构

```
├── app.py           # Flask 后端 + 检测内核 v4.0
├── index.html       # 前端单页应用
├── logo.ico         # 图标
├── .gitignore
└── README.md
```

## 许可证

MIT
