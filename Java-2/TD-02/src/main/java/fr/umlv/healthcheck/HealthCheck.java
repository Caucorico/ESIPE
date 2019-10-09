package fr.umlv.healthcheck;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Objects;
import java.util.Optional;

class HealthCheck {

    static boolean healthCheck( URI uri ) throws InterruptedException {
        var httpClient = HttpClient.newBuilder().build();
        var request = HttpRequest.newBuilder()
            .uri(uri)
            .timeout(Duration.ofMinutes(2))
            .build();

        try {
            var response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (IOException e) {
            return false;
        }
    }

    public static void main(String[] args) {
//        var test = healthCheck(URI.create("https://google.com"));
//        System.out.println(test);
    }

    @FunctionalInterface
    public interface URIFinder {
        Optional<URI> find();

        private static Optional<URI> optionalUriIfValid( String s ) {
            try {
                return Optional.of(new URI(s));
            } catch ( URISyntaxException e ) {
                return Optional.empty();
            }
        }

        static URIFinder fromArguments( String[] args ){
            Objects.requireNonNull(args);
            return () -> Optional.of(args).filter(a -> a.length > 0 ).flatMap( a -> optionalUriIfValid(a[0]));
        }

        static URIFinder fromURI(String uri) {
            Objects.requireNonNull(uri);
            return () -> optionalUriIfValid(uri);
        }

        default URIFinder or( URIFinder uriFinder ) {
            Objects.requireNonNull(uriFinder);
            return () -> find().or(uriFinder::find);
        }
    }

}
