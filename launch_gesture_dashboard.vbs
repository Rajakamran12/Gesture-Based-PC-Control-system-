Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

projectDir = FSO.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = projectDir & "\.venv\Scripts\pythonw.exe"
appPath = projectDir & "\gesture_pc_control\main.py"

If FSO.FileExists(pythonwPath) And FSO.FileExists(appPath) Then
    cmd = """" & pythonwPath & """ """" & appPath & """"
    WshShell.Run cmd, 0, False
Else
    WScript.Echo "Could not find pythonw or app file." & vbCrLf & _
                 "Expected:" & vbCrLf & pythonwPath & vbCrLf & appPath
End If
