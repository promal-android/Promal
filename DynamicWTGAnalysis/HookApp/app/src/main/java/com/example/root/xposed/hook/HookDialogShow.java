package com.example.root.xposed.hook;

import android.app.ActivityManager;
import android.app.AndroidAppHelper;
import android.app.Dialog;
import android.content.ComponentName;
import android.content.Context;
import android.view.View;
import android.widget.Adapter;
import android.widget.ListView;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedHelpers;
import de.robv.android.xposed.callbacks.XC_LoadPackage;

import static com.example.root.xposed.Utils.getOutputForLongString;
import static com.example.root.xposed.Utils.getStack;
import static com.example.root.xposed.Utils.getTimeStamp;
import static de.robv.android.xposed.XposedHelpers.findAndHookMethod;

public class HookDialogShow {

    private static List<View> removeNull(View[] mChildren){
        List<View> viewlist = new ArrayList<>();

        for(View v : mChildren) {
            if(v != null) {
                viewlist.add(v);
            }
        }
        return viewlist;
    }

    private static boolean hasField(View view){
        try {
            Field field = XposedHelpers.findField(view.getClass(), "mChildren");
            return true;
        }catch (NoSuchFieldError e) {
            return false;
        }
    }

    private static boolean hasChild(View view){
        if (hasField(view)){
            View[] mChildren = (View[]) XposedHelpers.getObjectField(view,"mChildren");
            List<View> mChildrens = removeNull(mChildren);
            if (mChildrens.size() > 0){
                return true;
            }else {
                return false;
            }
        }
        return false;
    }

    private static String getChild(View view, String result){
        try{
            if (hasChild(view)){
                View[] mChildren = (View[]) XposedHelpers.getObjectField(view,"mChildren");
                List<View> mChildrens = removeNull(mChildren);
                for (int i = 0; i < mChildrens.size(); i++) {
                    View childview = mChildrens.get(i);
                    int mViewFlags = (int) XposedHelpers.getObjectField(childview,"mViewFlags");
//                            0x0000000C is the View.VISIBILITY_MASK
                    if ((mViewFlags & 0x0000000C) == View.VISIBLE) {
                        String output = childview.toString();
//                        if (output.contains(":")) {
                        result = result + output + "\t";
                    }
                    if (hasChild(childview)){
                        result = getChild(childview, result);
                    }
                }
            }} catch (Error e){} catch (Exception e){}
        return result;
    }

    public static void init(final XC_LoadPackage.LoadPackageParam lpparam){
        try {
            findAndHookMethod(
                "android.app.Dialog",
                lpparam.classLoader,
                "show",
                new XC_MethodHook() {
                    @Override
                    protected void afterHookedMethod(MethodHookParam param)
                            throws Throwable {
                        String pkg = lpparam.packageName;
                        String TAG = getTimeStamp();

                        Dialog dialog = (Dialog) param.thisObject;

                        Context context = AndroidAppHelper.currentApplication();
                        ActivityManager am = (ActivityManager) context.getSystemService(context.ACTIVITY_SERVICE);
                        ComponentName cn = am.getRunningTasks(1).get(0).topActivity;

                        List outputs = new ArrayList();
                        outputs.add(pkg);
                        outputs.add("DialogShow");
                        outputs.add(TAG);
                        outputs.add("Dialog:" +dialog.toString());

                        List outputs_stack = new ArrayList();
                        outputs_stack.add(pkg);
                        outputs_stack.add("DialogShow");
                        outputs_stack.add(TAG);
                        outputs_stack.add("Dialog:" +dialog.toString());

                        List outputs_decor = new ArrayList();
                        outputs_decor.add(pkg);
                        outputs_decor.add("DialogShow");
                        outputs_decor.add(TAG);
                        outputs_decor.add("Dialog:" +dialog.toString());

                        outputs.add("SourceActivity:" + cn.getClassName());
                        try {
                            String title = (String) XposedHelpers.getObjectField(XposedHelpers.getObjectField(dialog,"mAlert"),"mTitle");
                            outputs.add("Title:"+title);
                        } catch (Error e){} catch (Exception e){}
                        try {
                            String title = (String) XposedHelpers.getObjectField(XposedHelpers.getObjectField(dialog,"title"),"mText");
                            outputs.add("Title:"+title);
                        } catch (Error e){} catch (Exception e){}
                        try {
                            ListView listview = (ListView) XposedHelpers.getObjectField(XposedHelpers.getObjectField(dialog,"mAlert"),"mListView");
                            Adapter adapter = listview.getAdapter();
                            outputs.add("List:"+XposedHelpers.getObjectField(adapter,"mObjects").toString());
                        } catch (Error e){} catch (Exception e){}

                        Throwable ex = new Throwable();
                        String[] stacks = getStack(ex);

                        String stackstr = "Stacks:";
                        for (int i = 0; i < stacks.length; i++) {
                            stackstr = stackstr + stacks[i] + "\t";
                        }
                        outputs_stack.add(stackstr.substring(0,stackstr.length()-1));

                        if (!stackstr.contains("ContextMenuBuilder")){
                            try {
                                View mDecor = (View) XposedHelpers.getObjectField(dialog,"mDecor");
                                String result = getChild(mDecor, "DecorView:" + mDecor.toString() + "\t");
                                outputs_decor.add(result.substring(0,result.length()-1));
                            } catch (Error e){} catch (Exception e){}
                            getOutputForLongString(outputs);
                            getOutputForLongString(outputs_stack);
                            getOutputForLongString(outputs_decor);
                        }
                    }
                });
        } catch (Error e){} catch (Exception e){}
    }
}
