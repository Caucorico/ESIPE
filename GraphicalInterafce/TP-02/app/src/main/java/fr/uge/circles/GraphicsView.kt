package fr.uge.circles

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.View
import fr.uge.circles.finger.Finger

class GraphicsView(context: Context, attrs: AttributeSet, defStyleAttr: Int) :
    View(context, attrs, defStyleAttr) {

    constructor(context: Context, attrs: AttributeSet) : this(context, attrs, 0) {}

    val paint: Paint = Paint()

    val fingers = Array<Finger>(10) { Finger() }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        paint.setColor(Color.BLACK)

        for ( finger in fingers ) {
            if ( finger.x != null && finger.y != null ) {
                canvas?.drawCircle(finger.x!!, finger.y!!, 50.0f, paint)
            }
        }
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {

        var count = event?.pointerCount

        when( event?.actionMasked ) {
            MotionEvent.ACTION_DOWN -> {
                Log.i("TEST", "ACTION_DOWN")
                fingers[0].x = event.getX()
                fingers[0].y = event.getY()
            }
            MotionEvent.ACTION_MOVE -> {
                if ( count != null ) {
                    count = count-1
                    for ( i in 0..count ) {
                        fingers[i].x = event.getX(i)
                        fingers[i].y = event.getY(i)
                    }
                }
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                if (count != null) {
                    fingers[event.actionIndex].x = event.getX(event.actionIndex)
                    fingers[event.actionIndex].y = event.getY(event.actionIndex)
                }
            }
        }
        invalidate()
        return true
    }

}