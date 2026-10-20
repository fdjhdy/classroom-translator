@echo off
cd /d D:\pyprogram\translator
rem API key is read from env var TRANSLATE_API_KEY (set it in System Properties > Environment Variables; never write it in this file)
set PATH=%USERPROFILE%\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cublas\bin;%USERPROFILE%\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cudnn\bin;%PATH%
"%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" server.py --subject "class" --translate-base-url https://api.siliconflow.cn/v1 --translate-model deepseek-ai/DeepSeek-V3.2
pause
