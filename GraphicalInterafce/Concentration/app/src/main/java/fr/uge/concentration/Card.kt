package fr.uge.concentration

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.IOException
import java.io.InputStream
import java.io.Serializable
import kotlin.collections.ArrayList


class Card(val name: String, var visible: Boolean = false): Serializable {

    companion object {
        lateinit var backBitmap: Bitmap

        fun loadCards(context: Context, path: String): List<Card> {
            /* Init the back */
            try {
                val input = context.resources.openRawResource(R.mipmap.card_back)
                backBitmap = BitmapFactory.decodeStream(input)
            } catch (e: IOException) {
                Log.e(Card::class.java.name, "back image ($path) not found")
            }

            /* Init the cards */
            val l: MutableList<Card> = ArrayList()
            try {
                for (filename in context.assets.list(path)!!) {
                    val name = filename.substring(0, filename.indexOf("."))
                    l.add(Card("$path/$filename"))
                    l.add(Card("$path/$filename"))
                    Log.v(Card::class.java.name, "image($path/$filename) loaded")
                }
            } catch (e: IOException) {
                Log.e(Card::class.java.name, e.message, e)
            }
            l.shuffle()

            Log.d(Card::class.java.name, l.toString())
            return l
        }
    }

    fun getBitMap(context: Context): Bitmap? {
        if ( !visible ) return backBitmap

        var inputStream: InputStream? = null
        try {
            inputStream = context.assets.open(name)
            return  BitmapFactory.decodeStream(inputStream)
        } catch (e: IOException) {
            try {
                inputStream?.close()
            } catch (e2: IOException) {
            }
        }

        return null
    }
}