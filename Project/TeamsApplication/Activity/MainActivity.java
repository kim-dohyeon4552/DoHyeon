package com.example.teamsapplication;


import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    Button startBtn1;
    Button startBtn2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        startBtn1 = findViewById(R.id.startBtn1);
        startBtn2 = findViewById(R.id.startBtn2);
        startBtn2.setClickable(true);
        startBtn1.setClickable(true);
        startBtn2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, GoogleLoginActivity.class );
                startActivity(intent);
            }
        });
        startBtn1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, GoogleLoginActivity.class  );
                startActivity(intent);
            }
        });
    }
}