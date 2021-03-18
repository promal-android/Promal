package com.example.root.xposed.communicationapp;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import com.example.root.xposed.Utils;

public class CommunicationService extends Service {
    public CommunicationService() {
    }

    @Override
    public IBinder onBind(Intent intent){
        return null;
    }

    public void onCreate() {
        super.onCreate();
        Log.i("Communication Service", "onCreate");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent !=null) {
            String line = intent.getStringExtra("DATA");
            if (line!=null) {
                Log.i("Communication Xposed", "Got data " + line);
                new Utils().execute(line);
            }
        }
        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.i("Communication Xposed", "onDestory");
    }
}
