package fr.uge.circles

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import fr.uge.circles.finger.Finger
import java.util.Random
import kotlin.collections.HashMap

class GraphicsView : View {

    constructor(context: Context, attrs: AttributeSet, defStyle: Int) : super(context, attrs, defStyle) {
        val r = Random()
        colors = IntArray(10) { Color.rgb(r.nextInt(255), r.nextInt(255), r.nextInt(255)) }
    }

    constructor(context: Context, attrs: AttributeSet) : this(context, attrs, 0) {}

    private val paint: Paint = Paint()

    private val colors: IntArray

    private val fingers = HashMap<Int, Finger>( 10 )

    /**
     * 1) Il existe deux façons d'ajouter un event onClick sur un objet View :
     * - On redéfinie la fonction onTouchEvent dans notre classe qui extends View
     * - On prend une instance de notre objet  et on fait un setOnTouchListener
     *
     * 2)
     * - La méthode onDraw est appelée lorsque androïd décide de rafraichir l'objet View.
     *   Cela peut survenir au démarrage, mais également lors de l'appel à la méthode invalidate().
     *
     * - La méthode invalidate() permet d'indiquer à Androïd qu'il faut redéssiner notre objet.
     * - La méthode requestLayout() demander de redessiner le layout entier plutôt que seulement
     *   notre objet. Ce n'est donc pas cette méthode que nous allons utiliser ici.
     *
     *
     *
     */
    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        var invalidate = false
        var time = 5.0f

        for ( (i, finger) in fingers.values.withIndex() ) {
            paint.color = colors[i]

            if ( finger.end == null ) {
                invalidate = true
                time = (System.currentTimeMillis() - finger.start).toFloat()
            } else {
                time = (finger.end!! - finger.start).toFloat()
            }

            canvas?.drawCircle(finger.x, finger.y, time/100.0f, paint)
        }

        if ( invalidate ) {
            invalidate()
        }
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        when( event?.actionMasked ) {
            MotionEvent.ACTION_DOWN -> {
                val actionIndex = event.actionIndex

                val finger = Finger(event.getX(actionIndex), event.getY(actionIndex), System.currentTimeMillis())
                fingers[event.getPointerId(actionIndex)] = finger
            }
            MotionEvent.ACTION_POINTER_DOWN  -> {
                val actionIndex = event.actionIndex

                val finger = Finger(event.getX(actionIndex), event.getY(actionIndex), System.currentTimeMillis())
                fingers[event.getPointerId(actionIndex)] = finger
            }
            MotionEvent.ACTION_MOVE -> {
                for ( i in 0 until event.pointerCount ) {
                    val finger = fingers[event.getPointerId(i)]
                    finger!!.x = event.getX(i)
                    finger.y = event.getY(i)

                    /* Problem here : Even immobile, the touchscreen detect movement... We are forced
                     * to refresh...
                     */
                    finger.end = System.currentTimeMillis()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> {
                val actionIndex = event.actionIndex
                val finger = fingers[event.getPointerId(actionIndex)]

                if ( finger != null && finger.end == null ) {
                    finger.end = System.currentTimeMillis()
                }
            }
        }
        invalidate()
        return true
    }

}