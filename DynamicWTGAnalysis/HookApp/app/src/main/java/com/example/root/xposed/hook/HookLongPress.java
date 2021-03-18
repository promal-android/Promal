package com.example.root.xposed.hook;

import android.view.View;

import com.example.root.xposed.Utils;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedHelpers;
import de.robv.android.xposed.callbacks.XC_LoadPackage;

public class HookLongPress {
    public static void init(final XC_LoadPackage.LoadPackageParam lpparam){
        final String pkg = lpparam.packageName;
        XposedHelpers.findAndHookMethod(
                "android.widget.AbsListView",
                lpparam.classLoader,
                "performLongPress",
                View.class,
                int.class,
                long.class,
                new XC_MethodHook() {

                    @Override
                    protected void afterHookedMethod(MethodHookParam param)
                            throws Throwable {
                        if (param.getResult().equals(true)){
                            Utils.outputLongClick((View)param.args[0],pkg);
                        }
                    }
                }
        );
        XposedHelpers.findAndHookMethod(
                "android.view.View",
                lpparam.classLoader,
                "performLongClick",
                new XC_MethodHook() {

                    @Override
                    protected void afterHookedMethod(MethodHookParam param)
                            throws Throwable {
                        if (param.getResult().equals(true)){
                            Utils.outputLongClick((View)param.thisObject,pkg);
                        }
                    }
                }
        );
    }
}
