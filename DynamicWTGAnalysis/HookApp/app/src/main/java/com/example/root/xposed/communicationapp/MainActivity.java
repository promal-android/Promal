package com.example.root.xposed.communicationapp;

import android.content.ComponentName;
import android.content.Intent;
import android.os.Bundle;

import android.view.View;

import android.widget.Button;

import android.widget.Toast;

import com.example.root.xposed.R;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private Button button;

    @Override

    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        button = (Button) findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {

            public void onClick(View v) {

                Intent serviceIntent = new Intent();
                serviceIntent.setComponent(new ComponentName("com.example.root.xposed", "com.example.root.xposed.testapp.CommunicationService"));
                serviceIntent.putExtra("DATA","Communication Service Started");
                startService(serviceIntent);

                Toast.makeText(MainActivity.this, toastMessage(), Toast.LENGTH_SHORT).show();

            }

        });

    }

    public String toastMessage() {

        return "This is first path";

    }

}