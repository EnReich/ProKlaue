@setlocal EnableDelayedExpansion enableextensions

@for /f "tokens=4-5 delims=[.XP " %%i in ('ver') do @(
	@set /a VERSION_MAJOR=%%i
	@set /a VERSION_MINOR=%%j
)

@set "VERSION_GEQ_6_2=F"
@if "!VERSION_MAJOR!" GTR "6" (
	@set "VERSION_GEQ_6_2=T"
)

@if "!VERSION_MAJOR!" EQU "6" (
	@if "!VERSION_MINOR!" GEQ "2" (
		@set "VERSION_GEQ_6_2=T"
	)
)

@SET "MAYAPY_DIR="
@for %%d in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do @(
	@for %%y in (2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029) do @(
		@if exist "%%d:\Program Files\Autodesk\Maya%%y\bin\mayapy.exe" (
			SET MAYAPY_DIR=%%d:\Program Files\Autodesk\Maya2019\bin
	)) 
)

@if "MAYAPY_DIR%"=="" (
	@echo Enter path to the directory where mayapy.exe is located: 
	set /p MAYAPY_DIR=
)

@for %%a in ("!MAYAPY_DIR!") do set "MAYAPY_PARENT_DIR=%%~dpa"

@SET MAYA_SITE_LIB_DIR=!MAYAPY_PARENT_DIR!\Python\Lib\site-packages

@if not exist "!MAYA_SITE_LIB_DIR!" (
	@mkdir "!MAYA_SITE_LIB_DIR!"
)

@SET PATH=%PATH%;".\..";"!MAYAPY_DIR!";"!MAYAPY_PARENT_DIR!\Python\Scripts"

@echo Copying packages to mayas site-packages

@if "!VERSION_GEQ_6_2!" EQU "T" (
	@echo Numpy...
	@robocopy ".\site-packages\numpy" "!MAYA_SITE_LIB_DIR!\numpy" /E /NFL /NDL /NJH /NJS /nc /ns /np /it
	@echo Scipy...
	@robocopy ".\site-packages\scipy" "!MAYA_SITE_LIB_DIR!\scipy" /E /NFL /NDL /NJH /NJS /nc /ns /np /it
	@echo Sklearn...
	@robocopy ".\site-packages\sklearn" "!MAYA_SITE_LIB_DIR!\sklearn" /E /NFL /NDL /NJH /NJS /nc /ns /np /it
) else (
	@echo Numpy...
	@xcopy ".\site-packages\numpy" "!MAYA_SITE_LIB_DIR!\numpy" /E 
	@echo Scipy...
	@xcopy ".\site-packages\scipy" "!MAYA_SITE_LIB_DIR!\scipy" /E 
	@echo Sklearn...
	@xcopy ".\site-packages\sklearn" "!MAYA_SITE_LIB_DIR!\sklearn" /E
)

@pause 
