package fr.uge.atomsyndicator.atom

import com.android.volley.Request
import com.android.volley.RequestQueue
import com.android.volley.Response
import com.android.volley.toolbox.StringRequest
import fr.uge.atomsyndicator.EntryAdapter

class AtomService {

    fun updateEntries(queue: RequestQueue, url: String, entries: ArrayList<AtomParser.Entry>, adapter: EntryAdapter) {
        // Request a string response from the provided URL.
        val stringRequest = StringRequest(
            Request.Method.GET, url,
            Response.Listener<String> { response ->
                AtomParser(response.reader()).parse(entries)
                adapter.notifyDataSetChanged()
            },
            Response.ErrorListener { entries.add(AtomParser.Entry(null, "That didn't work!", null, null, null)) })

        // Add the request to the RequestQueue.
        queue.add(stringRequest)
    }
}