package fr.uge.circles.finger

import java.sql.Time
import java.time.Instant

data class Finger(var x: Float, var y: Float, var start: Long = System.currentTimeMillis(), var end: Long? = null)