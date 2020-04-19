package fr.uge.concentration

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView

class CardAdapter(private val cards: List<Card>, private val itemClickListener: View.OnClickListener)
    : RecyclerView.Adapter<CardAdapter.ViewHolder>() {

    class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val image = itemView.findViewById<ImageView>(R.id.item_card_icon)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        val viewItem = inflater.inflate(R.layout.item_card, parent, false)
        return ViewHolder(viewItem)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val card = cards[position]
        holder.image.setImageBitmap(card.getBitMap(holder.itemView.context))
        holder.image.tag = position

        if ( !card.visible ) {
            holder.image.setOnClickListener(itemClickListener)
        }
    }

    override fun getItemCount(): Int {
        return cards.size
    }
}