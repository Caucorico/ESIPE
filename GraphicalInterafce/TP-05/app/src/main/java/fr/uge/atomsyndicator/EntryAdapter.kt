package fr.uge.atomsyndicator

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.cardview.widget.CardView
import androidx.recyclerview.widget.RecyclerView
import fr.uge.atomsyndicator.atom.AtomParser

class EntryAdapter(private val entries: ArrayList<AtomParser.Entry>, private val itemClickListener: View.OnClickListener)
    : RecyclerView.Adapter<EntryAdapter.ViewHolder>() {

    val dateService: DateService = DateService()

    class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val cardView = itemView.findViewById<CardView>(R.id.item_entry_card_view)
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
        holder.date.text = dateService.getFormatedDuration(entry.date)
        holder.resume.text = entry.summary

        holder.cardView.tag = position
        holder.cardView.setOnClickListener(itemClickListener)
    }

    override fun getItemCount(): Int {
        return entries.size
    }
}