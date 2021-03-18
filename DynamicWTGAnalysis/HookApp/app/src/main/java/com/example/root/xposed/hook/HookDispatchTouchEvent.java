package com.example.root.xposed.hook;


import android.app.ActivityManager;
import android.app.AndroidAppHelper;
import android.content.ComponentName;
import android.content.Context;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;

import com.example.root.xposed.Utils;

import java.util.ArrayList;
import java.util.List;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedHelpers;
import de.robv.android.xposed.callbacks.XC_LoadPackage;

import static com.example.root.xposed.Utils.getOutput;
import static com.example.root.xposed.Utils.getTimeStamp;
import static de.robv.android.xposed.XposedHelpers.findAndHookMethod;

public class HookDispatchTouchEvent {
    public static void init(final XC_LoadPackage.LoadPackageParam lpparam){

        try {
            findAndHookMethod(
                "android.view.View",
                lpparam.classLoader,
                "dispatchTouchEvent",
                MotionEvent.class,
                new XC_MethodHook() {

                    @Override
                    protected void afterHookedMethod(MethodHookParam param)
                            throws Throwable {
                        if ((Boolean) param.getResult()) {
                            final String pkg = lpparam.packageName;
                            final String TAG = getTimeStamp();
                            List outputs = new ArrayList();
                            outputs.add(pkg);
                            outputs.add("#");
                            outputs.add("TouchEvent:" + "View");
                            outputs.add(TAG);

                            View view = (View) param.thisObject;
                            final MotionEvent event = (MotionEvent) param.args[0];

                            if (event.getAction() == MotionEvent.ACTION_UP) {

                                //告诉PC端获取当前ViewTree和ScreenShot
                                Utils.saveCurrentState(pkg, TAG);

                                //获取TopActivity
                                Context context = AndroidAppHelper.currentApplication();
                                ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                                ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                                outputs.add("SourceActivity:" + cn.getClassName());

                                float currentX = event.getRawX();
                                float currentY = event.getRawY();
                                outputs.add("x:" + String.valueOf(currentX));
                                outputs.add("y:" + String.valueOf(currentY));

                                outputs.add("Widget:"+view.toString());

                                if (view.getId()!= -1) {
                                    outputs.add("WidgetID:" + context.getResources().getResourceName(view.getId()));
                                } else {
                                    outputs.add("WidgetID:NONE");
                                }

                                getOutput(outputs);
                            }
                        }
                    }
                });
        } catch (Error e){} catch (Exception e){}

        try {
            findAndHookMethod(
                    "android.app.Activity",
                    lpparam.classLoader,
                    "onTouchEvent",
                    MotionEvent.class,
                    new XC_MethodHook() {

                        @Override
                        protected void afterHookedMethod(MethodHookParam param)
                                throws Throwable {
                            if ((Boolean) param.getResult()) {
                                final String pkg = lpparam.packageName;
                                final String TAG = getTimeStamp();
                                List outputs = new ArrayList();
                                outputs.add(pkg);
                                outputs.add("#");
                                outputs.add("TouchEvent:" + "Activity");
                                outputs.add(TAG);

                                final MotionEvent event = (MotionEvent) param.args[0];

                                if (event.getAction() == MotionEvent.ACTION_UP) {

                                    //告诉PC端获取当前ViewTree和ScreenShot
                                    Utils.saveCurrentState(pkg, TAG);

                                    //获取TopActivity
                                    Context context = AndroidAppHelper.currentApplication();
                                    ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                                    ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                                    outputs.add("SourceActivity:" + cn.getClassName());

                                    float currentX = event.getRawX();
                                    float currentY = event.getRawY();
                                    outputs.add("x:" + String.valueOf(currentX));
                                    outputs.add("y:" + String.valueOf(currentY));

                                    getOutput(outputs);
                                }
                            }
                        }
                    });
        } catch (Error e){} catch (Exception e){}

        try {
            findAndHookMethod(
                    "android.view.ViewGroup",
                    lpparam.classLoader,
                    "dispatchTouchEvent",
                    MotionEvent.class,
                    new XC_MethodHook() {

                        @Override
                        protected void afterHookedMethod(MethodHookParam param)
                                throws Throwable {
                            try {
                                if ((Boolean) param.getResult()) {
                                    boolean is_intercepted = (Boolean) XposedHelpers.callMethod(param.thisObject, "onInterceptTouchEvent", param.args[0]);
                                    if (is_intercepted) {
//                                    if ((XposedHelpers.getObjectField(param.thisObject,"mFirstTouchTarget")) == null){
                                        final String pkg = lpparam.packageName;
                                        final String TAG = getTimeStamp();
                                        List outputs = new ArrayList();
                                        outputs.add(pkg);
                                        outputs.add("#");
                                        outputs.add("TouchEvent:" + "ViewGroup");
                                        outputs.add(TAG);

                                        ViewGroup view = (ViewGroup) param.thisObject;
                                        final MotionEvent event = (MotionEvent) param.args[0];

                                        //告诉PC端获取当前ViewTree和ScreenShot
                                        Utils.saveCurrentState(pkg, TAG);

                                        //获取TopActivity
                                        Context context = AndroidAppHelper.currentApplication();
                                        ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                                        ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                                        outputs.add("SourceActivity:" + cn.getClassName());

                                        float currentX = event.getRawX();
                                        float currentY = event.getRawY();
                                        outputs.add("x:" + String.valueOf(currentX));
                                        outputs.add("y:" + String.valueOf(currentY));

                                        outputs.add("Widget:"+view.toString());

                                        if (view.getId()!= -1) {
                                            outputs.add("WidgetID:" + context.getResources().getResourceName(view.getId()));
                                        } else {
                                            outputs.add("WidgetID:NONE");
                                        }

                                        getOutput(outputs);
                                    }
                                }
                            } catch (Error e){} catch (Exception e){}
                        }
                    });
        } catch (Error e){} catch (Exception e){}
    }
}
