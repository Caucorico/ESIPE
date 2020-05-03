package fr.uge.atomsyndicator

import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.android.volley.toolbox.Volley
import fr.uge.atomsyndicator.atom.AtomParser
import fr.uge.atomsyndicator.atom.AtomService


class FeedActivity : AppCompatActivity(), View.OnClickListener {

    private val entries = ArrayList<AtomParser.Entry>()

    private val atomService = AtomService()

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

            atomService.updateEntries(queue, url, entries, adapter)

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
            val orientation = this.resources.configuration.orientation

            val af = ArticleFragment()
            af.url = entry.url

            if ( orientation == Configuration.ORIENTATION_PORTRAIT ) {
                supportFragmentManager.beginTransaction()
                    .replace(R.id.activity_feed_container, af)
                    .addToBackStack(null)
                    .commit()
            } else if ( orientation == Configuration.ORIENTATION_LANDSCAPE ) {
                supportFragmentManager.beginTransaction()
                    .replace(R.id.activity_feed_article_fragment_container, af)
                    .addToBackStack(null)
                    .commit()
            }
        }
    }

    /* TODO : update the visible ViewHolder every minutes. */

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when(item.itemId) {
            R.id.main_menu_refresh -> {
                return true
            }
            R.id.main_menu_change_data_source -> {
                return true
            }
            else -> return super.onOptionsItemSelected(item)
        }
    }
}
