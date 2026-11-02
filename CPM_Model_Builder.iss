[Setup]
AppId={{56C55C1B-BC29-4FE2-8C9A-C55B5E2733A9}
AppName=CPM Model Builder
AppVersion=1.0.0
AppPublisher=Nahuel Soliz
AppSupportURL=https://www.linkedin.com/in/nahuelsoliz
DefaultDirName={autopf}\CPM Model Builder
DefaultGroupName=CPM Model Builder
OutputDir=installer
OutputBaseFilename=CPM_Model_Builder_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\CPM_Model_Builder.exe
SetupLogging=yes

[Files]
Source: "dist\CPM_Model_Builder\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CPM Model Builder"; Filename: "{app}\CPM_Model_Builder.exe"
Name: "{autodesktop}\CPM Model Builder"; Filename: "{app}\CPM_Model_Builder.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\CPM_Model_Builder.exe"; Description: "Launch CPM Model Builder"; Flags: postinstall nowait skipifsilent
