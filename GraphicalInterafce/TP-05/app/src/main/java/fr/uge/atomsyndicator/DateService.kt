package fr.uge.atomsyndicator

import java.time.Duration
import java.time.Instant
import java.util.*

class DateService {

    fun getFormatedDuration(date: Date): String {
        val duration = System.currentTimeMillis() - date.time
        var seconds = duration/1000
        var minutes = seconds/60
        var hours = minutes/60
        minutes %= 60
        seconds %= 60
        return "$hours heures, $minutes minutes et $seconds secondes"
    }
}