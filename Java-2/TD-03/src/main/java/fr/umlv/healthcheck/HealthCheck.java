package fr.umlv.healthcheck;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;
import java.util.function.Function;

public class HealthCheck {

    public static boolean healthCheck(URI uri) throws InterruptedException {
        HttpClient client = HttpClient.newBuilder().build();
        HttpRequest request = HttpRequest.newBuilder().uri(uri).build();
        try {
            HttpResponse response = client.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (IOException e) {
            return false;
        }
    }

    @FunctionalInterface
    public interface URIFinder {

        Optional<URI> find();

        static Optional<URI> newURI(String uri){
            if(uri == null)
                return Optional.empty();
            try {
                return Optional.of(new URI(uri));
            }catch(URISyntaxException e) {
                return Optional.empty();
            }
        }

        static URIFinder fromArguments(String[] args) {
            Objects.requireNonNull(args);
            return () -> Optional.of(args)
                .filter( arg -> arg.length != 0 )
                .flatMap( arg -> newURI(arg[0]) );
        }

        static URIFinder fromURI(String uri) {
            Objects.requireNonNull(uri);
            return () -> newURI(uri);
        }

        default URIFinder or(URIFinder finder) {
            Objects.requireNonNull(finder);
            return () -> find().or(finder::find);
        }

        static <T> URIFinder fromMapGetLike(T key, Function<? super T, String> function) {
            Objects.requireNonNull(key);
            Objects.requireNonNull(function);

            return fromURI(function.apply(key));
        }

        static URIFinder fromPropertyFile(Path path, String key) {
            Objects.requireNonNull(path);
            Objects.requireNonNull(key);

            try {
                var bufferedReader = Files.newBufferedReader(path);
                var properties = new Properties();
                properties.load(bufferedReader);
                return fromURI((String) properties.get(key));
            } catch (IOException e) {
                return Optional::empty;
            }
        }
    }
}