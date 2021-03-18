# DynamicWTGAnalysis

Dynamic WTG analysis first instruments apps with an Xposed module, then leverages app exploration techniques to automatically explore apps’ behaviors and collect the runtime information for building dynamic.

## HookApp

HookApp is an Xposed module that instruments an app to record the widget interactions and window transitions.  
To correctly record window transitions, the instrumentation records the source window, and the target window when a transition happens during runtime.  
During the explorations, the displays of dialog windows and menu windows (including both OptionsMenu and ContextMenu) are also recorded, together with the attributes of these windows such as titles, texts, and the call stacks of their parent methods, so that they can be accurately matched with the windows in static WTGs.  

## Usage of HookApp

0. Make sure your testing device is at API 23, rooted and [Xposed Framework](https://repo.xposed.info/module/de.robv.android.xposed.installer) is installed.
1. obtain the **onCreateOptionsMenu** and **onCreateContextMenu** methods within testing apps with **libdex.zip** and replace the **target.log** [here](https://github.com/ICSE2021Promal/promal/tree/master/DynamicWTGAnalysis/HookApp/app/src/main/assets).  
2. open the **HookApp** project with Android Studio, build&install it on your testing device, tick to activate the installed module and reboot.


## Paladin

Paladin is a state-of-the-art app exploration tool to trigger app behaviors.  
During the explorations, ProMal records the clicked UI widget in each interaction and also obtains the view tree and screenshot for each window.  
Since some UI widgets may not possess widget IDs, PROMAL uses coordinates and XPath to identify UI widgets in a window.  


## Usage of Paladin

0. Make sure your testing device is connected to your PC. The testing device and PC should be on the same LAN (for instance using the same WIFI).
1. Install **uiautomator.apk** and **uiautomator-androidTest.apk** on your testing device.
2. Run the **SocketServerPaladin.py** script on your PC.
3. Configure **config.json** accordingly. Specifically, **ADB_PATH** should be changed to the path of your Android SDK, **SERIAL** should be changed to the serial number of your testing device, **PACKAGE** should be changed to the package name of the app to be tested.  
4. Run **java -jar paladin.jar**, the view tree, screenshot and Xposed log will be recorded within the same dir of **SocketServerPaladin.py**.
