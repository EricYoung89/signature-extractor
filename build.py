# -*- coding: utf-8 -*-
"""打包成单文件 exe：.venv\\Scripts\\python.exe build.py

依赖只有 numpy、Pillow（运行）和 PyInstaller（打包）。下面排除的模块程序都用不到，
去掉它们能让 exe 从约 24MB 降到约 18MB。注意 numpy.ma、decimal 不能排除
（np.percentile 和 Pillow 的 PNG 插件会间接用到）。
"""

import os
import shutil

import PyInstaller.__main__

EXCLUDES = [
    "cv2", "matplotlib", "scipy", "pandas",
    # Pillow：AVIF 编解码（7.5MB）和字体渲染用不到
    "PIL._avif", "PIL.AvifImagePlugin", "PIL._imagingft",
    # numpy 用不到的子模块
    "numpy.random", "numpy.fft", "numpy.polynomial", "numpy.testing", "numpy.f2py", "numpy.distutils",
    # 标准库：网络/加密（会带上 OpenSSL）、测试调试、压缩、并发等
    "ssl", "_ssl", "_hashlib", "http", "email", "xmlrpc", "unicodedata",
    "unittest", "doctest", "pdb", "pydoc", "sqlite3", "_sqlite3",
    "lzma", "_lzma", "bz2", "_bz2", "multiprocessing", "asyncio", "concurrent",
]

here = os.path.dirname(os.path.abspath(__file__))
icon = os.path.join(here, "assets", "icon.ico")
args = [
    os.path.join(here, "signature_extractor.py"),
    "--noconfirm", "--onefile", "--windowed", "--name", "签名提取工具",
    "--icon", icon,                                   # exe 文件图标
    "--add-data", icon + os.pathsep + "assets",       # 窗口标题栏/任务栏图标
    "--distpath", os.path.join(here, "dist"),
    "--workpath", os.path.join(here, "build"),
    "--specpath", os.path.join(here, "build"),
]
for m in EXCLUDES:
    args += ["--exclude-module", m]

PyInstaller.__main__.run(args)
shutil.rmtree(os.path.join(here, "build"), ignore_errors=True)
