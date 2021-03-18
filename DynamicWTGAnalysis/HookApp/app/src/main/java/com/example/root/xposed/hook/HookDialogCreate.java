package com.example.root.xposed.hook;

import android.app.ActivityManager;
import android.app.AndroidAppHelper;
import android.app.Dialog;
import android.content.ComponentName;
import android.content.Context;

import java.util.ArrayList;
import java.util.List;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.callbacks.XC_LoadPackage;

import static com.example.root.xposed.Utils.getOutputForLongString;
import static com.example.root.xposed.Utils.getStack;
import static com.example.root.xposed.Utils.getTimeStamp;
import static de.robv.android.xposed.XposedHelpers.findAndHookConstructor;

public class HookDialogCreate {

    public static void init(final XC_LoadPackage.LoadPackageParam lpparam){
        try {
            findAndHookConstructor(
                "android.app.Dialog",
                lpparam.classLoader,
                Context.class,
                int.class,
                boolean.class,
                new XC_MethodHook() {
                    @Override
                    protected void afterHookedMethod(MethodHookParam param)
                            throws Throwable {
                        String pkg = lpparam.packageName;
                        String TAG = getTimeStamp();

                        Dialog dialog = (Dialog) param.thisObject;
                        List outputs = new ArrayList();

                        Context context = AndroidAppHelper.currentApplication();
                        ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                        ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                        outputs.add(pkg);
                        outputs.add("DialogConstructor");
                        outputs.add(TAG);
                        outputs.add("Dialog:" + dialog.toString());

                        List outputs_stack = new ArrayList();
                        outputs_stack.add(pkg);
                        outputs_stack.add("DialogConstructor");
                        outputs_stack.add(TAG);
                        outputs_stack.add("Dialog:" + dialog.toString());

                        outputs.add("SourceActivity:" + cn.getClassName());

                        Throwable ex = new Throwable();
                        String[] stacks = getStack(ex);

                        String stackstr = "Stacks:";
                        for (int i = 0; i < stacks.length; i++) {
                            stackstr = stackstr + stacks[i] + "\t";
                        }
                        outputs_stack.add(stackstr.substring(0,stackstr.length()-1));

                        getOutputForLongString(outputs);
                        getOutputForLongString(outputs_stack);

                    }
                });
        } catch (Error e){} catch (Exception e){}
    }
}
