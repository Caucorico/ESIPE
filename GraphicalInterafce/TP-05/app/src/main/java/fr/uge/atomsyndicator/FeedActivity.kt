package fr.uge.atomsyndicator

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.android.volley.Request
import com.android.volley.Response
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import fr.uge.atomsyndicator.atom.AtomParser
import java.util.logging.Logger

class FeedActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_feed)

        val recyclerView = findViewById<RecyclerView>(R.id.activity_feed_recycler_view)

        // Instantiate the RequestQueue.
        val queue = Volley.newRequestQueue(this)
        val url = "http://android-developers.blogspot.com/feeds/posts/default"
        val entries = ArrayList<AtomParser.Entry>()
        val adapter = EntryAdapter(entries)

        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        // Request a string response from the provided URL.
        val stringRequest = StringRequest(Request.Method.GET, url,
                Response.Listener<String> { response ->
                    AtomParser(response.reader()).parse(entries)
                    adapter.notifyDataSetChanged()
                },
                Response.ErrorListener { entries.add(AtomParser.Entry(null, "That didn't work!", null, null, null)) })

        // Add the request to the RequestQueue.
        queue.add(stringRequest)

    }
}
