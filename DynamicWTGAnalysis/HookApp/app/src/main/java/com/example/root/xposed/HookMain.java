package com.example.root.xposed;

import de.robv.android.xposed.IXposedHookLoadPackage;
import de.robv.android.xposed.XposedBridge;
import de.robv.android.xposed.callbacks.XC_LoadPackage.LoadPackageParam;

import com.example.root.xposed.hook.HookBothMenus;
import com.example.root.xposed.hook.HookDialogCreate;
import com.example.root.xposed.hook.HookDispatchTouchEvent;
import com.example.root.xposed.hook.HookDialogShow;
import com.example.root.xposed.hook.HookKeyEvent;
import com.example.root.xposed.hook.HookLongPress;

public class HookMain implements IXposedHookLoadPackage {

    @Override
    public void handleLoadPackage(final LoadPackageParam lpparam) {

        String pkg = lpparam.packageName;

        // Hook keyevent operation (BACK and MENU)
        try {
            if (lpparam.packageName.equals("android")) {
                HookKeyEvent.init(lpparam);
            }
        } catch (Error e){}catch (Exception e){}

        if (!Utils.PRE_INSTALLED.contains("package:"+pkg+"\n")) {
            XposedBridge.log("Hook Loaded App:" + pkg);
            HookDispatchTouchEvent.init(lpparam);
            HookLongPress.init(lpparam);
            HookBothMenus.init(lpparam);
            HookDialogShow.init(lpparam);
            HookDialogCreate.init(lpparam);
            }
    }
}
