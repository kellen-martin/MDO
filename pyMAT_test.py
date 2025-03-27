import sys
from os import path

apistartup = "api_startup_file.txt"
GmatInstall = "C:\\Users\\weasd\\Desktop\\GMAT\\gmat-win-R2022a\\GMAT"
GmatBinPath = GmatInstall + "/bin"
Startup = GmatBinPath + "/" + apistartup

if path.exists(Startup):
    print(f'Running GMAT in {GmatInstall}')

    sys.path.insert(1, GmatBinPath)

    import gmatpy as gmat

    gmat.Setup(Startup)

else:
    print("Cannot find ", Startup)
    print()
    print("Please set up a GMAT startup file named ", apistartup, " in the ",
          GmatBinPath, " folder.")