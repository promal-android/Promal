package com.example.root.xposed;

import android.app.AndroidAppHelper;
import android.content.ComponentName;

import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.util.Log;
import android.view.View;

import java.io.OutputStream;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

import de.robv.android.xposed.XposedBridge;

public class Utils extends AsyncTask<String, Integer, String> {

    //Change String ip to your PC
    static public String ip = "10.0.3.2";
    static public int port = 9998;
    // 0 don't save view tree, 1 save view tree to PC
    static public int saveViewTree = 1;
    // 0 output to XposedLog, 1 to PC, 2 to both
    static public int outputType = 2;

    static public String packageName = "com.example.root.xposed";
    static public String className = "com.example.root.xposed.communicationapp.CommunicationService";

    //Exclude pre_installed apps when hooking, copied from "adb shell pm list package"
    public static final String PRE_INSTALLED =
            "package:android\n" +
            "package:com.amaze.filemanager\n" +
            "package:com.andrew.apollo\n" +
            "package:com.android.apps.tag\n" +
            "package:com.android.backupconfirm\n" +
            "package:com.android.bluetooth\n" +
            "package:com.android.bluetoothmidiservice\n" +
            "package:com.android.browser\n" +
            "package:com.android.calculator2\n" +
            "package:com.android.calendar\n" +
            "package:com.android.calllogbackup\n" +
            "package:com.android.camera2\n" +
            "package:com.android.camera\n" +
            "package:com.android.captiveportallogin\n" +
            "package:com.android.carrierconfig\n" +
            "package:com.android.cellbroadcastreceiver\n" +
            "package:com.android.certinstaller\n" +
            "package:com.android.contacts\n" +
            "package:com.android.customlocale2\n" +
            "package:com.android.defcontainer\n" +
            "package:com.android.deskclock\n" +
            "package:com.android.development\n" +
            "package:com.android.development_settings\n" +
            "package:com.android.dialer\n" +
            "package:com.android.documentsui\n" +
            "package:com.android.dreams.basic\n" +
            "package:com.android.dreams.phototable\n" +
            "package:com.android.email\n" +
            "package:com.android.exchange\n" +
            "package:com.android.externalstorage\n" +
            "package:com.android.frameworks.telresources\n" +
            "package:com.android.galaxy4\n" +
            "package:com.android.gallery3d\n" +
            "package:com.android.gesture.builder\n" +
            "package:com.android.hotwordenrollment\n" +
            "package:com.android.htmlviewer\n" +
            "package:com.android.incallui\n" +
            "package:com.android.inputdevices\n" +
            "package:com.android.inputmethod.latin\n" +
            "package:com.android.keychain\n" +
            "package:com.android.keyguard\n" +
            "package:com.android.launcher3\n" +
            "package:com.android.launcher\n" +
            "package:com.android.location.fused\n" +
            "package:com.android.magicsmoke\n" +
            "package:com.android.managedprovisioning\n" +
            "package:com.android.messaging\n" +
            "package:com.android.mms.service\n" +
            "package:com.android.mms\n" +
            "package:com.android.music\n" +
            "package:com.android.musicfx\n" +
            "package:com.android.musicvis\n" +
            "package:com.android.nfc\n" +
            "package:com.android.noisefield\n" +
            "package:com.android.omadm.service\n" +
            "package:com.android.onetimeinitializer\n" +
            "package:com.android.packageinstaller\n" +
            "package:com.android.pacprocessor\n" +
            "package:com.android.phasebeam\n" +
            "package:com.android.phone\n" +
            "package:com.android.printspooler\n" +
            "package:com.android.providers.calendar\n" +
            "package:com.android.providers.calllogbackup\n" +
            "package:com.android.providers.contacts\n" +
            "package:com.android.providers.downloads.ui\n" +
            "package:com.android.providers.downloads\n" +
            "package:com.android.providers.media\n" +
            "package:com.android.providers.settings\n" +
            "package:com.android.providers.telephony\n" +
            "package:com.android.providers.userdictionary\n" +
            "package:com.android.provision\n" +
            "package:com.android.proxyhandler\n" +
            "package:com.android.quicksearchbox\n" +
            "package:com.android.sdm.plugins.connmo\n" +
            "package:com.android.sdm.plugins.dcmo\n" +
            "package:com.android.sdm.plugins.diagmon\n" +
            "package:com.android.sdm.plugins.sprintdm\n" +
            "package:com.android.server.telecom\n" +
            "package:com.android.settings\n" +
            "package:com.android.sharedstoragebackup\n" +
            "package:com.android.shell\n" +
            "package:com.android.smspush\n" +
            "package:com.android.soundrecorder\n" +
            "package:com.android.statementservice\n" +
            "package:com.android.stk\n" +
            "package:com.android.systemui\n" +
            "package:com.android.terminal\n" +
            "package:com.android.vending\n" +
            "package:com.android.videoeditor\n" +
            "package:com.android.voicedialer\n" +
            "package:com.android.vpndialogs\n" +
            "package:com.android.wallpaper.holospiral\n" +
            "package:com.android.wallpaper.livepicker\n" +
            "package:com.android.wallpaper\n" +
            "package:com.android.wallpapercropper\n" +
            "package:com.android.webview\n" +
            "package:com.bel.android.dspmanager\n" +
            "package:com.cyanogenmod.account\n" +
            "package:com.cyanogenmod.eleven\n" +
            "package:com.cyanogenmod.filemanager\n" +
            "package:com.cyanogenmod.lockclock\n" +
            "package:com.cyanogenmod.setupwizard\n" +
            "package:com.cyanogenmod.trebuchet\n" +
            "package:com.cyanogenmod.updater\n" +
            "package:com.cyanogenmod.wallpapers\n" +
            "package:com.example.android.apis\n" +
            "package:com.example.android.livecubes\n" +
            "package:com.example.root.xposed\n" +
            "package:com.genymotion.genyd\n" +
            "package:com.genymotion.superuser\n" +
            "package:com.genymotion.systempatcher\n" +
            "package:com.github.uiautomator.test\n" +
            "package:com.github.uiautomator\n" +
            "package:com.google.android.apps.gcs\n" +
            "package:com.google.android.apps.tycho\n" +
            "package:com.google.android.backuptransport\n" +
            "package:com.google.android.feedback\n" +
            "package:com.google.android.gms\n" +
            "package:com.google.android.gsf.login\n" +
            "package:com.google.android.gsf\n" +
            "package:com.google.android.launcher.layouts.genymotion\n" +
            "package:com.google.android.onetimeinitializer\n" +
            "package:com.google.android.partnersetup\n" +
            "package:com.google.android.play.games\n" +
            "package:com.google.android.setupwizard\n" +
            "package:com.google.android.syncadapters.calendar\n" +
            "package:com.google.android.syncadapters.contacts\n" +
            "package:com.google.language\n" +
            "package:com.google.system.sensor\n" +
            "package:com.lge.HiddenMenu\n" +
            "package:com.lge.SprintHiddenMenu\n" +
            "package:com.lge.lifetimer\n" +
            "package:com.lge.update\n" +
            "package:com.lr.keyguarddisabler\n" +
            "package:com.qti.qualcomm.datastatusnotification\n" +
            "package:com.qualcomm.atfwd\n" +
            "package:com.qualcomm.qcrilmsgtunnel\n" +
            "package:com.qualcomm.qti.rcsbootstraputil\n" +
            "package:com.qualcomm.qti.rcsimsbootstraputil\n" +
            "package:com.qualcomm.shutdownlistner\n" +
            "package:com.qualcomm.timeservice\n" +
            "package:com.quicinc.cne.CNEService\n" +
            "package:com.redbend.vdmc\n" +
            "package:com.speedsoftware.rootexplorer\n" +
            "package:com.svox.pico\n" +
            "package:com.verizon.omadm\n" +
            "package:cyanogenmod.platform\n" +
            "package:de.robv.android.xposed.installer\n" +
            "package:eu.chainfire.supersu\n" +
            "package:io.github.ylimit.droidbotapp\n" +
            "package:jackpal.androidterm\n" +
            "package:jp.co.omronsoft.openwnn\n" +
            "package:net.cactii.flash2\n" +
            "package:org.codeaurora.bluetooth\n" +
            "package:org.codeaurora.ims\n" +
            "package:org.cyanogenmod.audiofx\n" +
            "package:org.cyanogenmod.bugreport\n" +
            "package:org.cyanogenmod.cmaudio.service\n" +
            "package:org.cyanogenmod.cmsettings\n" +
            "package:org.cyanogenmod.gello.browser\n" +
            "package:org.cyanogenmod.hexolibre\n" +
            "package:org.cyanogenmod.launcher.home\n" +
            "package:org.cyanogenmod.livelockscreen.service\n" +
            "package:org.cyanogenmod.profiles\n" +
            "package:org.cyanogenmod.providers.datausage\n" +
            "package:org.cyanogenmod.screencast\n" +
            "package:org.cyanogenmod.snap\n" +
            "package:org.cyanogenmod.theme.chooser2\n" +
            "package:org.cyanogenmod.theme.chooser\n" +
            "package:org.cyanogenmod.themes.provider\n" +
            "package:org.cyanogenmod.themeservice\n" +
            "package:org.cyanogenmod.voiceplus\n" +
            "package:org.cyanogenmod.wallpaperpicker\n" +
            "package:org.cyanogenmod.wallpapers.photophase\n" +
            "package:org.cyanogenmod.weather.provider\n" +
            "package:org.cyanogenmod.weatherservice\n" +
            "package:org.whispersystems.whisperpush\n"+
            "package:system\n";

    @Override
    protected String doInBackground(String... strings) {
        String line = strings[0];
        writeToSocket(line);
        return "Done";
    }

    @Override
    protected void onPostExecute(String result) {
        // after finish the task
    }

    public static void writeToLog(String strLine) {
        if (outputType == 0) {
            String tag = "#";
            XposedBridge.log(tag + strLine);
        } else if (outputType == 1){
            Utils.writeToPC(strLine);
        } else{
            String tag = "#";
            XposedBridge.log(tag + strLine);
            Utils.writeToPC(strLine);
        }
    }

    public static void writeToLogForLongString(String strLine) {
        if (outputType == 0) {
            String tag = "#";
            XposedBridge.log(tag + strLine);
        } else if (outputType == 1){
            Utils.writeToSocketThread(strLine);
        } else{
            String tag = "#";
            XposedBridge.log(tag + strLine);
            Utils.writeToSocketThread(strLine);
        }
    }

    public static void writeToPC(String strLine){

        Intent intent = new Intent();

        intent.setComponent(new ComponentName(packageName,className));
        intent.putExtra("DATA", strLine);

        Context context = AndroidAppHelper.currentApplication();
        context.startService(intent);

    }

    public static void writeToPCWithContext(String strLine, Context context){

        Intent intent = new Intent();

        intent.setComponent(new ComponentName(packageName,className));
        intent.putExtra("DATA", strLine);

        context.startService(intent);

    }

    public static void saveCurrentState(String pkg, String TAG){
        if (saveViewTree == 0) return ;
        Intent intent = new Intent();

        intent.setComponent(new ComponentName(packageName, className));
        intent.putExtra("DATA","SaveCurrentState"+","+pkg+","+TAG);

        Context context = AndroidAppHelper.currentApplication();
        context.startService(intent);
    }

    public static void writeToSocketThread(final String strLine){

        new Thread(new Runnable(){
            @Override
            public void run() {
                writeToSocket(strLine);
            }
        }).start();
    }

    public static void writeToSocket(String strLine){
        Socket sock;
        try {
            sock = new Socket(ip, port);
            // Send strLine to server
            OutputStream os = sock.getOutputStream();
            Log.i("Xposed","Sending...." + strLine);
            byte[] byteArray = strLine.getBytes();
            os.write(byteArray);
            os.flush();
            sock.close();
        } catch (Error e){} catch (Exception e){}
    }

    public static void getOutputForLongString(List outputs){
        String Output = "";
        for (int i = 0; i < outputs.size(); i++) {
            Output = Output + outputs.get(i) + "\n";
        }
        writeToLogForLongString(Output);
    }


    public static void getOutput(List outputs){
        String Output = "";
        for (int i = 0; i < outputs.size(); i++) {
            Output = Output + outputs.get(i) + "\n";
        }
        writeToLog(Output);
    }

    public static String getTimeStamp(){
        return String.valueOf(System.currentTimeMillis()/1000);
    }

    public static String[] getStack(Throwable ex) {
        StackTraceElement[] stackElements = ex.getStackTrace();
        String[] stacks = {};
        if (stackElements.length > 1) {
            stacks = new String[stackElements.length - 2];
            for (int i = 2; i < stackElements.length; i++) {
                String classnm = stackElements[i].getClassName() + ':' + stackElements[i].getMethodName() + ':' + stackElements[i].getLineNumber();
                stacks[i - 2] = classnm;
            }
        }
        return stacks;
    }

    public static void outputLongClick(View view, String pkg){
        String TAG = getTimeStamp();
        List outputs = new ArrayList();
        outputs.add(pkg);
        outputs.add("LongClick");
        outputs.add(TAG);
        outputs.add("Widget:"+view.toString());

        Context context = AndroidAppHelper.currentApplication();
        if (view.getId()!= -1) {
            outputs.add("WidgetID:" + context.getResources().getResourceName(view.getId()));
        } else {
            outputs.add("WidgetID:NONE");
        }
        getOutput(outputs);
    }

}
