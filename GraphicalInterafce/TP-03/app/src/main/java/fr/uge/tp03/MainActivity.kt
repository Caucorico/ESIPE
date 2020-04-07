package fr.uge.tp03

import android.media.MediaPlayer
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import java.io.Serializable

data class State(var start: Long = 0, var started: Boolean = false, var end: Long = 0, var time: Long = 0): Serializable

class MainActivity : AppCompatActivity() {

    lateinit var state: State
    lateinit var textView: TextView
    lateinit var editText: EditText
    lateinit var button: Button
    lateinit var handler: Handler
    val REFRESH: Long = 10

    val runnable = Runnable {
        val diff: Long = System.currentTimeMillis() - state.start
        val diffS = diff/1000
        val diffM = diff%1000
        val contentText = "$diffS : $diffM"

        textView.text = contentText
        Log.v("MainActivity", contentText)

        if ( (System.currentTimeMillis()-state.start)/1000 >= 5  ) {
            hideChrono()
        }

        handler.postDelayed(getIncreamentRunnable(), REFRESH)
    }

    private fun getIncreamentRunnable(): Runnable {
        return runnable
    }

    private fun hideChrono() {
        textView.setCompoundDrawablesWithIntrinsicBounds(
            R.drawable.fish, 0, 0, 0
        )
    }

    private fun initDisplay() {
        if ( state.started ) {
            textView.visibility = View.VISIBLE
            editText.visibility = View.GONE
            button.text = "Stop"
        } else {
            textView.visibility = View.GONE
            editText.visibility = View.VISIBLE
            button.text = "Start"
        }
    }

    private fun manageEnd() {
        textView.setCompoundDrawablesWithIntrinsicBounds(0, 0, 0, 0)
        val mediaPlayer: MediaPlayer?

        if ( state.end > state.start + state.time*1000*0.5 &&  state.end < state.start + state.time*1000*1.5 ) {
            textView.setText("Victory")
            if ( state.end > state.start + state.time*1000*0.95 &&  state.end < state.start + state.time*1000*1.05 ) {
                mediaPlayer = MediaPlayer.create(this, R.raw.sound1)
            } else if ( state.end > state.start + state.time*1000*0.85 &&  state.end < state.start + state.time*1000*1.15 ) {
                mediaPlayer = MediaPlayer.create(this, R.raw.sound2)
            } else if ( state.end > state.start + state.time*1000*0.7 &&  state.end < state.start + state.time*1000*1.3 ) {
                mediaPlayer = MediaPlayer.create(this, R.raw.sound3)
            } else {
                mediaPlayer = MediaPlayer.create(this, R.raw.sound4)
            }

            mediaPlayer.start()
        } else {
            textView.setText("Loose")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        handler = Handler()
        textView = findViewById(R.id.activity_main_timer)
        editText = findViewById(R.id.activity_main_edit_text)
        button = findViewById(R.id.activity_main_button)

        if ( savedInstanceState != null ) {
            Log.i("MainActivity", "State recover from bundle !")
            state = savedInstanceState.getSerializable("state") as State
        } else {
            Log.i("MainActivity", "New State !")
            state = State()
        }

        button.setOnClickListener {
            if (!state.started) {
                state.start = System.currentTimeMillis()
                state.started = true
                state.time = editText.text.toString().toLong()
                button.text = "Stop"
                textView.visibility = View.VISIBLE
                editText.visibility = View.GONE

                runnable.run()
            } else {
                state.end = System.currentTimeMillis()
                state.started = false
                handler.removeCallbacks(runnable)
                manageEnd()
            }

        }

        initDisplay()
    }

    override fun onStart() {
        super.onStart()

        if ( state.started ) {
            Log.i("MainActivity", "The chrono was already started, I rerun it !")
            handler.postDelayed(runnable, REFRESH)
        }
    }

    override fun onStop() {
        super.onStop()
        Log.i("MainActivity", "The code is now sleeping.")
        handler.removeCallbacks(runnable)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putSerializable("state", state)
    }
}
