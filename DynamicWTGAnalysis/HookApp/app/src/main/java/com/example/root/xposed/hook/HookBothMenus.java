package com.example.root.xposed.hook;

import android.app.ActivityManager;
import android.app.AndroidAppHelper;
import android.app.Application;
import android.content.ComponentName;
import android.content.Context;
import android.content.res.AssetManager;
import android.view.ContextMenu;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

import com.example.root.xposed.BuildConfig;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedHelpers;
import de.robv.android.xposed.callbacks.XC_LoadPackage;

import static com.example.root.xposed.Utils.getOutput;
import static com.example.root.xposed.Utils.getTimeStamp;
import static com.example.root.xposed.Utils.writeToPCWithContext;

public class HookBothMenus {

    public static void init(final XC_LoadPackage.LoadPackageParam lpparam){
        XposedHelpers.findAndHookMethod(Application.class, "attach", Context.class, new XC_MethodHook() {
            @Override
            protected void afterHookedMethod(MethodHookParam param) throws Throwable {
                Context context = (Context) param.args[0];
                String pkg = lpparam.packageName;
                String TAG = getTimeStamp();
//                捕获app启动
                writeToPCWithContext("SaveCurrentState-Start"+","+pkg+","+TAG, context);
                try {
                    AssetManager assetManager = context.createPackageContext(BuildConfig.APPLICATION_ID, Context.CONTEXT_IGNORE_SECURITY).getResources().getAssets();
                    InputStreamReader inputReader = new InputStreamReader(assetManager.open("target.log"));
                    BufferedReader bufferedReader = new BufferedReader(inputReader);
                    bufferedReader.readLine();
                    String line;
                    while ((line = bufferedReader.readLine()) != null) {
                        String[] strings = line.split(":");
                        String callerClass = strings[0];
                        final String callerMethod = strings[1];
                        if (callerMethod.equals("onCreateOptionsMenu")){
                            try {
                                XposedHelpers.findAndHookMethod(
                                        callerClass,
                                        lpparam.classLoader,
                                        callerMethod,
                                        Menu.class,
                                        new XC_MethodHook() {
                                            @Override
                                            protected void afterHookedMethod(MethodHookParam param)
                                                    throws Throwable {
                                                List titles = new ArrayList();
                                                Menu menu = (Menu) param.args[0];
                                                try {
                                                    ArrayList items = (ArrayList) XposedHelpers.getObjectField(menu,"mItems");
                                                    for (int i = 0; i < items.size(); i++) {
                                                        int mShowAsAction = (int) XposedHelpers.getObjectField(items.get(i),"mShowAsAction");
                                                        if (mShowAsAction == 0){
                                                            String title = (String) XposedHelpers.getObjectField(items.get(i),"mTitle");
                                                            titles.add(title);
                                                        }
                                                    }
                                                }catch (Error e){} catch (Exception e){}

                                                String pkg = lpparam.packageName;
                                                String TAG = getTimeStamp();
                                                List outputs = new ArrayList();
                                                outputs.add(pkg);
                                                outputs.add("OptionsMenu");
                                                outputs.add(TAG);

                                                Context context = AndroidAppHelper.currentApplication();
                                                ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                                                ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                                                outputs.add("SourceActivity:" + cn.getClassName());
                                                outputs.add("List:"+titles.toString());

                                                getOutput(outputs);
                                            }
                                        });

                            } catch (Error e){} catch (Exception e){}
                        } else if (callerMethod.equals("onCreateContextMenu")){
                            try {
                                XposedHelpers.findAndHookMethod(
                                        callerClass,
                                        lpparam.classLoader,
                                        callerMethod,
                                        ContextMenu.class,
                                        View.class,
                                        ContextMenu.ContextMenuInfo.class,
                                        new XC_MethodHook() {
                                            @Override
                                            protected void afterHookedMethod(MethodHookParam param)
                                                    throws Throwable {
                                                List titles = new ArrayList();
                                                ContextMenu menu = (ContextMenu) param.args[0];
                                                List outputs = new ArrayList();
                                                View v = (View) param.args[1];
                                                String pkg = lpparam.packageName;
                                                String TAG = getTimeStamp();
                                                outputs.add(pkg);
                                                outputs.add("ContextMenu");
                                                outputs.add(TAG);

                                                Context context = AndroidAppHelper.currentApplication();
                                                ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                                                ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                                                outputs.add("SourceActivity:" + cn.getClassName());
                                                outputs.add("Widget:"+v.toString());

                                                if (v.getId()!= -1) {
                                                    outputs.add("WidgetID:" + context.getResources().getResourceName(v.getId()));
                                                } else {
                                                    outputs.add("WidgetID:NONE");
                                                }
                                                try {
                                                    outputs.add("Title:"+XposedHelpers.getObjectField(menu,"mHeaderTitle"));
                                                }catch (Error e){} catch (Exception e){}
                                                try {
                                                    ArrayList items = (ArrayList) XposedHelpers.getObjectField(menu,"mItems");
                                                    for (int i = 0; i < items.size(); i++) {
                                                        String title = (String) XposedHelpers.getObjectField(items.get(i),"mTitle");
                                                        titles.add(title);
                                                    }
                                                }catch (Error e){} catch (Exception e){}
                                                outputs.add("List:"+titles.toString());

                                                getOutput(outputs);
                                            }
                                        });

                            } catch (Error e){} catch (Exception e){}
                        } else if (callerMethod.equals("onOptionsItemSelected")){
                            try {
                                XposedHelpers.findAndHookMethod(
                                        callerClass,
                                        lpparam.classLoader,
                                        callerMethod,
                                        MenuItem.class,
                                        new XC_MethodHook() {
                                            @Override
                                            protected void afterHookedMethod(MethodHookParam param)
                                                    throws Throwable {
                                                MenuItem item = (MenuItem) param.args[0];
                                                List outputs = new ArrayList();
                                                String pkg = lpparam.packageName;
                                                String TAG = getTimeStamp();
                                                outputs.add(pkg);
                                                outputs.add("MenuItem");
                                                outputs.add(TAG);

                                                Context context = AndroidAppHelper.currentApplication();
                                                ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                                                ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                                                outputs.add("SourceActivity:" + cn.getClassName());

                                                try {
                                                    outputs.add("MenuItemID:" + context.getResources().getResourceName(item.getItemId()));
                                                } catch (Exception e) {
                                                    outputs.add("MenuItemID:NONE");
                                                }

                                                getOutput(outputs);
                                            }
                                        });

                            } catch (Error e){} catch (Exception e){}
                        }
                    }
                } catch (Error e){} catch (Exception e){}
            }});
    }
}