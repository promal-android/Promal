package com.example.root.xposed.hook;

import android.app.ActivityManager;
import android.app.AndroidAppHelper;
import android.content.ComponentName;
import android.content.Context;
import android.view.KeyEvent;

import com.example.root.xposed.Utils;

import java.util.ArrayList;
import java.util.List;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.callbacks.XC_LoadPackage;
import de.robv.android.xposed.callbacks.XCallback;

import static com.example.root.xposed.Utils.getOutput;
import static com.example.root.xposed.Utils.getTimeStamp;
import static de.robv.android.xposed.XposedHelpers.findAndHookMethod;

public class HookKeyEvent {

	public static void init(final XC_LoadPackage.LoadPackageParam lpparam) {

		try {
			findAndHookMethod(
//					"com.android.internal.policy.impl.PhoneWindowManager", //for android 4.4
					"com.android.server.policy.PhoneWindowManager", //for android 6.0+
					lpparam.classLoader,
					"interceptKeyBeforeQueueing",
					KeyEvent.class,
					int.class,
//					boolean.class,
					new XC_MethodHook(XCallback.PRIORITY_HIGHEST) {
						@Override
						protected void beforeHookedMethod(MethodHookParam param) throws Throwable {
							KeyEvent ie = (KeyEvent) param.args[0];
							String eventstr = String.valueOf(ie);
							try {
								String TAG = getTimeStamp();
								List outputs = new ArrayList();

								if (eventstr.contains("keyCode=KEYCODE_BACK")) {
									if (eventstr.contains("action=ACTION_DOWN")) {
										Context context = AndroidAppHelper.currentApplication();
										ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
										ComponentName cn = am.getRunningTasks(1).get(0).topActivity;
										String pkg = cn.getPackageName();
										if (!Utils.PRE_INSTALLED.contains("package:"+pkg+"\n")) {
											Utils.saveCurrentState(pkg,TAG);
											outputs.add(pkg);
											outputs.add("#");
											outputs.add(TAG);
											outputs.add("Event:BACK");
											outputs.add("SourceActivity:" + cn.getClassName());
											getOutput(outputs);
										}
									}
								}
								else if (eventstr.contains("keyCode=KEYCODE_MENU")) {
									if (eventstr.contains("action=ACTION_DOWN")) {
										Context context = AndroidAppHelper.currentApplication();
										ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
										ComponentName cn = am.getRunningTasks(1).get(0).topActivity;
										String pkg = cn.getPackageName();
										if (!Utils.PRE_INSTALLED.contains("package:"+pkg+"\n")) {
											Utils.saveCurrentState(pkg,TAG);
											outputs.add(pkg);
											outputs.add("#");
											outputs.add(TAG);
											outputs.add("Event:MENU");
											outputs.add("SourceActivity:" + cn.getClassName());
											getOutput(outputs);
										}
									}
								}
							} catch (Error e){} catch (Exception e){}
						}
					});
		} catch (Error e){} catch (Exception e){}
	}
}
