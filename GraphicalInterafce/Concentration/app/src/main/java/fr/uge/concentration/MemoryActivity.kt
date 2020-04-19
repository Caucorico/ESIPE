package fr.uge.concentration

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.Serializable

class MemoryActivity : AppCompatActivity(), View.OnClickListener {

    private lateinit var adapter: CardAdapter
    private lateinit var state: State

    data class State(val liste: List<Card>, var firstSelection: Card? = null,
                     var secondSelection: Card? = null, var pairFound: Int = 0, var turn: Int = 0,
                     var start: Long = 0): Serializable

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_memory)

        if ( savedInstanceState != null ) {
            state = savedInstanceState.getSerializable("state") as State
        } else {
            state = State(Card.loadCards(this, "classical"))
        }

        adapter = CardAdapter(state.liste, this)

        val recyclerView = findViewById<RecyclerView>(R.id.activity_memory_recycler_view)
        recyclerView.layoutManager = GridLayoutManager(this, 4)
        recyclerView.adapter = adapter
    }

    override fun onClick(v: View) {
        if (v.tag == null) return
        val index = v.tag as Int
        val card = state.liste[index]

        if ( state.start == 0L ) state.start = System.currentTimeMillis()

        if ( state.firstSelection == null ) {
            state.firstSelection = card
            Log.d(MemoryActivity::class.java.name, "First selection null")
        } else if ( state.secondSelection == null ) {
            state.secondSelection = card
            Log.d(MemoryActivity::class.java.name, "Second selection null")
            if ( state.firstSelection?.name == state.secondSelection?.name ) {
                state.pairFound++
            }
        } else {
            Log.d(MemoryActivity::class.java.name, "Nothing null")

            if ( state.firstSelection?.name != state.secondSelection?.name ) {
                state.firstSelection?.visible = false
                state.secondSelection?.visible = false

                adapter.notifyDataSetChanged()
            }

            state.turn++
            state.firstSelection = card
            state.secondSelection = null
        }

        card.visible = true
        adapter.notifyItemChanged(index)

        Log.i(MemoryActivity::class.java.name, "Avancement : ${state.pairFound}/${state.liste.size/2}")
        if ( state.pairFound == /*state.liste.size/2*/1 ) {
            val intent = Intent(this, VictoryActivity::class.java)
            intent.putExtra("data", VictoryData(state.turn, state.start))
            startActivity(intent)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putSerializable("state", state)
    }
}
