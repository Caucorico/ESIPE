package fr.uge.clickgame

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.Toast

class ClickGameActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_click_game)

        val button01:Button = findViewById(R.id.button01)

        button01.setOnClickListener{
            val toast: Toast = Toast.makeText(this, R.string._activity_click_game_quit, Toast.LENGTH_SHORT)
            toast.show()
        }
    }


}
