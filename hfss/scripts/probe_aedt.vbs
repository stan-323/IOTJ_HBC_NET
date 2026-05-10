Dim fso, shell, rootDir, scriptDir, outDir, outFile
Set shell = CreateObject("WScript.Shell")
rootDir = shell.Environment("PROCESS")("HBC_HFSS_ROOT")
Set fso = CreateObject("Scripting.FileSystemObject")
If rootDir = "" Then
    scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
    rootDir = fso.GetParentFolderName(scriptDir)
End If
outDir = rootDir & "\outputs"
If Not fso.FolderExists(outDir) Then
    fso.CreateFolder(outDir)
End If
outFile = outDir & "\aedt_probe_vbs.txt"

Dim app, desktop, file
Set app = CreateObject("Ansoft.ElectronicsDesktop.2024.1")
Set desktop = app.GetAppDesktop()
Set file = fso.CreateTextFile(outFile, True)
file.WriteLine "AEDT VBS initialized"
file.WriteLine "version=" & desktop.GetVersion()
file.Close
desktop.QuitApplication
