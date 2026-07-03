@echo off
chcp 65001 >nul
title RoboOS - Galbot-1 清理桌面任务

echo ============================================================
echo   RoboOS + Galbot-1  一键启动脚本
echo   启动顺序: Redis ^> skill.py ^> Master ^> Slaver
echo ============================================================
echo.

:: ── 配置区：按实际情况修改 ───────────────────────────────────────
:: RoboOS 项目根目录
set ROBOOS_DIR=%~dp0

:: Python 解释器（如果用 conda，改成 conda run -n 你的环境名 python）
set PYTHON=python

:: Redis 启动方式（已在 PATH 中则直接用，否则填绝对路径）
set REDIS=redis-server

:: conda 环境名（如果用 conda 管理环境，取消下面两行注释并填写）
:: set CONDA_ENV=roboos
:: set PYTHON=conda run -n %CONDA_ENV% python
:: ─────────────────────────────────────────────────────────────────

echo [1/4] 启动 Redis...
start "Redis" cmd /k "%REDIS%"
timeout /t 2 /nobreak >nul
echo       Redis 已启动（端口 6379）
echo.

echo [2/4] 启动 skill.py（Galbot-1 MCP 工具服务，端口 8000）...
start "skill.py" cmd /k "cd /d %ROBOOS_DIR%slaver\galbot-1 && %PYTHON% skill.py"
echo       等待 skill.py 就绪...
timeout /t 5 /nobreak >nul
echo       skill.py 已启动
echo.

echo [3/4] 启动 Master（任务规划，端口 5000）...
start "Master" cmd /k "cd /d %ROBOOS_DIR%master && %PYTHON% run.py"
echo       等待 Master 就绪...
timeout /t 5 /nobreak >nul
echo       Master 已启动
echo.

echo [4/4] 启动 Slaver（工具调用代理）...
start "Slaver" cmd /k "cd /d %ROBOOS_DIR%slaver && %PYTHON% run.py"
echo       等待 Slaver 注册到 Master...
timeout /t 8 /nobreak >nul
echo       Slaver 已启动
echo.

echo ============================================================
echo   所有服务已启动！
echo.
echo   【发送任务】在新终端运行：
echo     python %ROBOOS_DIR%test\send_task.py
echo   或直接 curl：
echo     curl -X POST http://localhost:5000/publish_task ^
echo          -H "Content-Type: application/json" ^
echo          -d "{\"task\": \"Clear the table: pick up the trash bag from the floor, bag the large items on the table, fetch the towel via replay trajectory, then use the towel to sweep lobster debris into the box and bag it.\"}"
echo.
echo   【真机上需要手动启动】（在机器人终端执行）：
echo     python camera_viewer_1.py
echo     python robot_server_clear_table.py --model-host ^<工作站IP^>
echo.
echo   【停止所有服务】直接关闭各终端窗口
echo ============================================================
echo.
pause
