package fr.uge.atomsyndicator

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.fragment.app.Fragment

class ArticleFragment() : Fragment() {

    var url: String? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        val view = inflater.inflate(R.layout.article_fragment, container, false)

        if ( url != null ) {
            val webView = view.findViewById<WebView>(R.id.article_fragment_web_view)
            val webViewClient: WebViewClient = object: WebViewClient() {}
            webView.webViewClient = webViewClient
            webView.loadUrl(url)
        }

        return view
    }

}