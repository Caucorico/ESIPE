package fr.uge.concentration

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity


class VictoryActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_victory)

        val score = findViewById<TextView>(R.id.activity_victory_score)
        val time = findViewById<TextView>(R.id.activity_victory_time)
        val restart = findViewById<Button>(R.id.activity_victory_restart)

        val data = intent.getSerializableExtra("data") as VictoryData?
        if ( data != null ) {
            val sec = (System.currentTimeMillis() - data.start)/1000
            score.text = "${data.score} turns"
            time.text = "$sec seconds"
        }
    }
}
