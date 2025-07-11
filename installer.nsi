; LottoProphet Windows Installer Script
; NSIS (Nullsoft Scriptable Install System) script for creating Windows installer

!include "MUI2.nsh"
!include "FileFunc.nsh"

; 定义应用程序名称和版本
!define APPNAME "LottoProphet"
!define APPVERSION "1.0.0"
!define PUBLISHER "Yang Zhao"
!define WEBSITE "https://github.com/zhaoyangpp/LottoProphet"

; 定义安装程序名称
OutFile "dist\${APPNAME}-${APPVERSION}-Setup.exe"

; 默认安装目录
InstallDir "$PROGRAMFILES\${APPNAME}"

; 获取安装目录的注册表项
InstallDirRegKey HKCU "Software\${APPNAME}" ""

; 请求管理员权限
RequestExecutionLevel admin

; 设置压缩选项
SetCompressor /SOLID lzma

; 现代界面设置
!define MUI_ABORTWARNING
!define MUI_ICON "dist\icon.ico"
!define MUI_UNICON "dist\icon.ico"

; 安装界面页面
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "dist\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; 卸载界面页面
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; 语言设置
!insertmacro MUI_LANGUAGE "SimpChinese"
!insertmacro MUI_LANGUAGE "English"

; 安装程序区段
Section "MainSection" SEC01
  SetOutPath "$INSTDIR"
  
  ; 添加主程序文件
  File "dist\${APPNAME}.exe"
  
  ; 添加数据文件和模型文件
  File /r "dist\data"
  File /r "dist\model"
  File /r "dist\update"
  
  ; 添加文档
  File "dist\README.md"
  File "dist\README_EN.md"
  File "dist\requirements.txt"
  
  ; 创建卸载程序
  WriteUninstaller "$INSTDIR\uninstall.exe"
  
  ; 创建开始菜单快捷方式
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\${APPNAME}.exe"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\卸载 ${APPNAME}.lnk" "$INSTDIR\uninstall.exe"
  
  ; 创建桌面快捷方式
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\${APPNAME}.exe"
  
  ; 写入注册表信息
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${APPVERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLInfoAbout" "${WEBSITE}"
  
  ; 获取安装大小并写入注册表
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "EstimatedSize" "$0"
  
  ; 保存安装目录到注册表
  WriteRegStr HKCU "Software\${APPNAME}" "" $INSTDIR
SectionEnd

; 卸载程序区段
Section "Uninstall"
  ; 删除程序文件
  Delete "$INSTDIR\${APPNAME}.exe"
  Delete "$INSTDIR\uninstall.exe"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\README_EN.md"
  Delete "$INSTDIR\requirements.txt"
  Delete "$INSTDIR\LICENSE"
  
  ; 删除数据目录和文件
  RMDir /r "$INSTDIR\data"
  RMDir /r "$INSTDIR\model"
  RMDir /r "$INSTDIR\update"
  
  ; 删除安装目录
  RMDir "$INSTDIR"
  
  ; 删除开始菜单快捷方式
  Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}\卸载 ${APPNAME}.lnk"
  RMDir "$SMPROGRAMS\${APPNAME}"
  
  ; 删除桌面快捷方式
  Delete "$DESKTOP\${APPNAME}.lnk"
  
  ; 删除注册表项
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
  DeleteRegKey HKCU "Software\${APPNAME}"
SectionEnd