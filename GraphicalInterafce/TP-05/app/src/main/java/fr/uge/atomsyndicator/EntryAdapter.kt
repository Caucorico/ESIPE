package fr.uge.atomsyndicator

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import fr.uge.atomsyndicator.atom.AtomParser

class EntryAdapter(val entries: ArrayList<AtomParser.Entry>) : RecyclerView.Adapter<EntryAdapter.ViewHolder>() {
    class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val title = itemView.findViewById<TextView>(R.id.item_entry_title)
        val date = itemView.findViewById<TextView>(R.id.item_entry_date)
        val resume = itemView.findViewById<TextView>(R.id.item_entry_resume)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        val viewItem = inflater.inflate(R.layout.item_entry, parent, false)
        return ViewHolder(viewItem)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val entry = entries[position]

        holder.title.text = entry.title
        holder.date.text = entry.date.toString()
        holder.resume.text = entry.summary


    }

    override fun getItemCount(): Int {
        return entries.size
    }
}