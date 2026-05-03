@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  台股 MA20 掃描 + 部署
echo ============================================================

:: 執行掃描（產生 index.html / margin_cache.json / tdcc_cache.json / Excel）
set PYTHONIOENCODING=utf-8
python scan.py
if errorlevel 1 (
    echo [錯誤] scan.py 執行失敗，中止部署
    pause
    exit /b 1
)

:: Git：新增 → commit → pull(以本地為準) → push
git add index.html margin_cache.json tdcc_cache.json

git diff --cached --quiet
if errorlevel 1 (
    for /f "tokens=*" %%d in ('powershell -command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
    git commit -m "auto: 更新台股 MA20 掃描結果 %TODAY%"
    git pull --rebase -X ours origin master
    git push
    echo [完成] 已推送至 GitHub Pages
) else (
    echo [跳過] 檔案無異動，不需 commit
)

echo.
echo 網頁：https://yoshiagent.github.io/tw-weekly-ma20/
pause
