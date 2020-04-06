package fr.uge.tp03

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.widget.Button
import android.widget.TextView
import java.io.Serializable

data class State(var start: Long = 0, var started: Boolean = false): Serializable

class MainActivity : AppCompatActivity() {

    lateinit var state: State
    lateinit var textView: TextView
    lateinit var button: Button
    lateinit var handler: Handler
    val REFRESH: Long = 10

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        handler = Handler()
        textView = findViewById(R.id.activity_main_timer)
        button = findViewById(R.id.activity_main_button)

        if ( savedInstanceState != null ) {
            Log.i("MainActivity", "State recover from bundle !")
            state = savedInstanceState.getSerializable("state") as State
        } else {
            Log.i("MainActivity", "New State !")
            state = State()
        }

        button.setOnClickListener {
            state.start = System.currentTimeMillis()
            state.started = true
            getRunnable().run()
        }
    }

    fun getRunnable(): Runnable {
        return Runnable {
            val diff: Long = System.currentTimeMillis() - state.start
            val diffS = diff/1000
            val diffM = diff%1000
            val contentText = "$diffS : $diffM"

            textView.text = contentText

            handler.postDelayed(getRunnable(), REFRESH)
        }
    }

    override fun onStart() {
        super.onStart()

        if ( state.started ) {
            Log.i("MainActivity", "The chrono was already started, I rerun it !")
            handler.postDelayed(getRunnable(), REFRESH)
        }
    }

    override fun onStop() {
        super.onStop()
        handler.removeCallbacks(getRunnable())
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putSerializable("state", state)
    }
}
