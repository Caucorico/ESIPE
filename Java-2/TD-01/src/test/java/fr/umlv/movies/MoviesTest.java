package fr.umlv.movies;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

@SuppressWarnings("static-method")
public class MoviesTest {
	@Test
	public void actorsByMovie() throws IOException {
		var path = Path.of("movies.txt");
		var actorsByMovie = Movies.actorsByMovie(path);
		
		var expectedMovies = List.of(
				Map.entry("Corps perdus (1990)", List.of("Tchéky Karyo", "Laura Morante")),
				Map.entry("Vampires: Los Muertos (2002)", List.of("Jon Bon Jovi", "Tim Guinee", "Arly Jover")),
				Map.entry("Clerks: Sell Out (2002)", List.of("Jeff Anderson", "Jason Mewes", "Brian O'Halloran", "Kevin Smith")),
				Map.entry("Punchdrunk Knuckle Love (2002)", List.of("Philip Seymour Hoffman", "Adam Sandler", "Emily Watson")),
				Map.entry("Spirit: Stallion of the Cimarron (2002)", List.of("Matt Damon"))
				);
		expectedMovies.forEach(entry -> assertEquals(entry.getValue(), actorsByMovie.get(entry.getKey())));
	}

	@Test
	public void numberOfUniqueActors() throws IOException {
		var path = Path.of("movies.txt");
		var actorsByMovie = Movies.actorsByMovie(path);
		var numberOfUniqueActors = Movies.numberOfUniqueActors(actorsByMovie);
		
		assertEquals(170509, numberOfUniqueActors);
	}

	@Test
	public void numberOfMoviesByActor() throws IOException {
		var path = Path.of("movies.txt");
		var actorsByMovie = Movies.actorsByMovie(path);
		var numberOfMoviesByActor = Movies.numberOfMoviesByActor(actorsByMovie);
		
		var expectedActorCouples = List.of(
						Map.entry("Emily Watson", 9L),
						Map.entry("Kevin Smith", 13L),
						Map.entry("Adam Sandler", 17L),
						Map.entry("Matt Damon", 21L),
						Map.entry("Tchéky Karyo", 22L)
					);
		expectedActorCouples.forEach(entry -> assertEquals(entry.getValue(), numberOfMoviesByActor.get(entry.getKey())));
	}

	@Test
	public void actorInMostMovies() throws IOException {
		var path = Path.of("movies.txt");
		var actorsByMovie = Movies.actorsByMovie(path);
		var numberOfMoviesByActor = Movies.numberOfMoviesByActor(actorsByMovie);
		var actorInMostMovies = Movies.actorInMostMovies(numberOfMoviesByActor).orElseThrow();
		
		assertAll(
		  () ->assertEquals("Frank Welker", actorInMostMovies.getKey()),
		  () ->assertEquals(92, actorInMostMovies.getValue())
		  );
	}
}
