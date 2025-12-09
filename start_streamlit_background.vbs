' Start Streamlit in background without visible command window
Set WshShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Change to the app directory
strPath = "C:\Users\NickK\OneDrive\Documents\stock alert app"
WshShell.CurrentDirectory = strPath

' Run Streamlit in hidden window (0 = hidden, 1 = normal, 2 = minimized)
WshShell.Run "cmd /c streamlit run Home.py", 0, False

' Optional: Show a notification that it started
MsgBox "Streamlit app started in background!" & vbCrLf & vbCrLf & _
       "Access it at: http://localhost:8501" & vbCrLf & vbCrLf & _
       "To stop it, use Task Manager to end the streamlit.exe process.", _
       vbInformation, "Stock Alert App"