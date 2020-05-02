package fr.uge.atomsyndicator

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.android.volley.Request
import com.android.volley.Response
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import fr.uge.atomsyndicator.atom.AtomParser


class FeedActivity : AppCompatActivity(), View.OnClickListener {

    private val entries = ArrayList<AtomParser.Entry>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_feed)

        val recyclerView = findViewById<RecyclerView>(R.id.activity_feed_recycler_view)
        val adapter = EntryAdapter(entries, this)
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        if ( savedInstanceState != null ) {
            entries.addAll(savedInstanceState.getSerializable("entries") as ArrayList<AtomParser.Entry>)
            Log.i(FeedActivity::class.java.name, "Entries list recover")
        } else {
            // Instantiate the RequestQueue.
            val queue = Volley.newRequestQueue(this)
            val url = "http://android-developers.blogspot.com/feeds/posts/default"

            // Request a string response from the provided URL.
            val stringRequest = StringRequest(Request.Method.GET, url,
                Response.Listener<String> { response ->
                    AtomParser(response.reader()).parse(entries)
                    adapter.notifyDataSetChanged()
                },
                Response.ErrorListener { entries.add(AtomParser.Entry(null, "That didn't work!", null, null, null)) })

            // Add the request to the RequestQueue.
            queue.add(stringRequest)
            Log.i(FeedActivity::class.java.name, "Entries list will be getted by Volley request")

            /* TODO : treat the case where the request is not finished before the rotation. */
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putSerializable("entries", entries)
    }

    override fun onClick(v: View) {
        if ( v.tag != null ) {
            val index = v.tag as Int
            val entry = entries[index]

            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(entry.url)))
        }
    }

    /* TODO : update the visible ViewHolder every minutes. */
}
