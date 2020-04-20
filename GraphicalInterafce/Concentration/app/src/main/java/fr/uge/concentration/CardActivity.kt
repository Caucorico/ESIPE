package fr.uge.concentration

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView

class CardActivity : AppCompatActivity() {

    private lateinit var liste: List<Card>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_card)
        liste = Card.loadCards(this, "classical")
    }

    override fun onStart() {
        super.onStart()
        val card = liste.random()
        val image = findViewById<ImageView>(R.id.activity_card_image_view)
        val textView = findViewById<TextView>(R.id.activity_card_text_view)

        image.setImageBitmap(card.getBitMap(this))
        textView.text = card.name
    }
}
