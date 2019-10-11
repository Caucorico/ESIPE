package fr.umlv.healthcheck;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import fr.umlv.healthcheck.HealthCheck.URIFinder;

@SuppressWarnings("static-method")
public class HealthCheckTest {
	@Test @Tag("Q1")
	public void healthCheckShouldWorkWithGoogleDotFr() throws Exception {
		var uri = URI.create("http://www.google.fr");
		assertTrue(HealthCheck.healthCheck(uri));
	}
	@Test @Tag("Q1")
	public void healthCheckShouldNotTryToRecoverInterruptedException() {
		var uri = URI.create("http://www.google.fr");
		try {
		  assertTrue(HealthCheck.healthCheck(uri));
		} catch(@SuppressWarnings("unused") InterruptedException e) {
			fail("interrupted exception");
		}
	}
	@Test @Tag("Q1")
	public void healthCheckShouldNotWorkWithWithTheWrongURIScheme() throws Exception {
		var uri = URI.create("ftp://www.google.fr");
		assertThrows(IllegalArgumentException.class, () -> HealthCheck.healthCheck(uri));
	}
	@Test @Tag("Q1")
	public void healthCheckWithNullShouldThrowANPE() throws Exception {
		assertThrows(NullPointerException.class, () -> HealthCheck.healthCheck(null));
	}
	
	
	@Test @Tag("Q1")
	public void healthCheckShouldWorkWithUPEM() throws InterruptedException {
		var uri = URI.create("http://www.u-pem.fr");
		assertTrue(HealthCheck.healthCheck(uri));
	}
	
	
	@Test @Tag("Q2")
	public void canCreateAURIFinderFromAMethodReference() {
		URIFinder alwaysEmpty = Optional::empty;
		assertTrue(alwaysEmpty.find().isEmpty());
	}
	
	
	@Test @Tag("Q3")
	public void uriFinderFromArgumentsWithGoogleDotFR() {
		var uriFinder = URIFinder.fromArguments(new String[] { "http://www.google.fr" });
		assertAll(
				() -> assertTrue(uriFinder.find().isPresent()),
			  () -> assertEquals(URI.create("http://www.google.fr"), uriFinder.find().orElseThrow())
			  );
	}
	@Test @Tag("Q3")
	public void uriFinderFromArgumentsWithUPEM() {
		var uriFinder = URIFinder.fromArguments(new String[] { "http://www.u-pem.fr" });
		assertAll(
				() -> assertTrue(uriFinder.find().isPresent()),
			  () -> assertEquals(URI.create("http://www.u-pem.fr"), uriFinder.find().orElseThrow())
			  );
	}
	@Test @Tag("Q3")
	public void uriFinderFromAnEmptyArgumentShouldBeEmpty() {
		var uriFinder = URIFinder.fromArguments(new String[0]);
		assertTrue(uriFinder.find().isEmpty());
	}
	@Test @Tag("Q3")
	public void uriFinderFromArgumentsNullShouldThrowANPE() {
		assertThrows(NullPointerException.class, () -> URIFinder.fromArguments(null));
	}
	
	
	@Test @Tag("Q4")
	public void uriFinderFromURIWithUPEM() {
		var uriFinder = URIFinder.fromURI("http://www.u-pem.fr");
		assertAll(
			() -> assertTrue(uriFinder.find().isPresent()),
		  () -> assertEquals(URI.create("http://www.u-pem.fr"), uriFinder.find().orElseThrow())
		  );
	}
	@Test @Tag("Q4")
	public void uriFinderFromURIWithGoogleFR() {
		var uriFinder = URIFinder.fromURI("http://www.google.fr");
		assertAll(
			() -> assertTrue(uriFinder.find().isPresent()),
		  () -> assertEquals(URI.create("http://www.google.fr"), uriFinder.find().orElseThrow())
		  );
	}
	@Test @Tag("Q4")
	public void uriFinderFromURIWithAnInvalidURI() {
		var uriFinder = URIFinder.fromURI("this is an invalid URI");
		assertTrue(uriFinder.find().isEmpty());
	}
	@Test @Tag("Q4")
	public void uriFinderFromArgumentsWithAnInvalidURI() {
		var uriFinder = URIFinder.fromArguments(new String[] { "this is an invalid URI" });
		assertTrue(uriFinder.find().isEmpty());
	}
	@Test @Tag("Q4")
	public void uriFinderFromURINullShouldThrowANPE() {
		assertThrows(NullPointerException.class, () -> URIFinder.fromURI(null));
	}
	
	
	@Test @Tag("Q5")
	public void uriFinderOrShouldTryTheFirstURIFinderFirst() {
		var uriFinder = URIFinder.fromURI("http://www.google.fr")
				.or(URIFinder.fromURI("http://www.u-pem.fr"));
		assertAll(
				() -> assertTrue(uriFinder.find().isPresent()),
			  () -> assertEquals(URI.create("http://www.google.fr"), uriFinder.find().orElseThrow())
			  );
	}
	@Test @Tag("Q5")
	public void uriFinderOrShouldUseTheSecondURIFinderIfTheFirstURIFinderIsEmpty() {
		URIFinder alwaysEmpty = Optional::empty;
		var uriFinder = alwaysEmpty
				.or(URIFinder.fromURI("http://www.u-pem.fr"));
		assertAll(
				() -> assertTrue(uriFinder.find().isPresent()),
			  () -> assertEquals(URI.create("http://www.u-pem.fr"), uriFinder.find().orElseThrow())
			  );
	}
	@Test @Tag("Q5")
	public void uriFinderOrNullShouldThrowANPE() {
		URIFinder alwaysEmpty = Optional::empty;
		assertThrows(NullPointerException.class, () -> alwaysEmpty.or(null));
	}
	
	
	@Test @Tag("Q6")
	public void uriFinderFromMapGetLikeShouldGetTheURIOfAnExistingKey() {
		var map = Map.of("1", "http://www.google.fr", "2", "http://www.u-pem.fr");
		var uriFinder = URIFinder.fromMapGetLike("1", map::get);
		assertAll(
				() -> assertTrue(uriFinder.find().isPresent()),
			  () -> assertEquals(URI.create("http://www.google.fr"), uriFinder.find().orElseThrow())
			  );
	}
	@Test @Tag("Q6")
	public void uriFinderFromMapGetLikeShouldNotGetTheURIOfAnNonExistingKey() {
		var map = Map.of("1", "http://www.google.fr", "2", "http://www.u-pem.fr");
		var uriFinder = URIFinder.fromMapGetLike("3", map::get);
		assertTrue(uriFinder.find().isEmpty());
	}
	@Test @Tag("Q6")
	public void uriFinderFromMapGetLikeShouldBeEmptyIfTheURIIsInvalid() {
		var map = Map.of("foo", "this is not a valid URI");
		var uriFinder = URIFinder.fromMapGetLike("foo", map::get);
		assertTrue(uriFinder.find().isEmpty());
	}
	@Test @Tag("Q6")
	public void uriFinderFromMapGetLikeNullShouldThrowANPE() {
		assertAll(
		  () -> assertThrows(NullPointerException.class, () -> URIFinder.fromMapGetLike(null, __ -> null)),
		  () -> assertThrows(NullPointerException.class, () -> URIFinder.fromMapGetLike("foo", null))
		  );
	}
	

	@Test @Tag("Q7")
	public void uriFinderFromMapGetLikeShouldGetTheURIOfAnExistingKeyEvenAsAnInteger() {
		var map = Map.of(1, "http://www.google.fr", 2, "http://www.u-pem.fr");
		var uriFinder = URIFinder.fromMapGetLike(1, map::get);
		assertAll(
				() -> assertTrue(uriFinder.find().isPresent()),
			  () -> assertEquals(URI.create("http://www.google.fr"), uriFinder.find().orElseThrow())
			  );
	}
	@Test @Tag("Q7")
	public void uriFinderFromMapGetLikeSignature() {
		// those four lines should compile
        URIFinder.fromMapGetLike(42, (Integer value) -> null);
        URIFinder.fromMapGetLike(LocalDate.now(), (LocalDate date) -> null);
        URIFinder.fromMapGetLike(42, (Object value) -> null);
        URIFinder.<Integer>fromMapGetLike(42, (Object value) -> null);
	}
	
	@Test @Tag("Q8")
	public void uriFinderFromPropertyFileWithTheRightKeyShouldBePresent() throws IOException {
		var path = Files.createTempFile("", "healthcheck-present.txt");
		Files.write(path, List.of("uri: http://www.google.fr", ""), UTF_8);
		try {
			var uriFinder = URIFinder.fromPropertyFile(path, "uri");
			assertAll(
					() -> assertTrue(uriFinder.find().isPresent()),
					() -> assertEquals(URI.create("http://www.google.fr"), uriFinder.find().orElseThrow())
					);
		} finally {
		  Files.delete(path);
		}
	}
	@Test @Tag("Q8")
	public void uriFinderFromPropertyFileWithAnInvalidURIShouldBeEmpty() throws IOException {
		var path = Files.createTempFile("", "healthcheck-invalid-uri.txt");
		Files.write(path, List.of("uri: this is not a valid URI", ""), UTF_8);
		try {
			var uriFinder = URIFinder.fromPropertyFile(path, "uri");
			assertTrue(uriFinder.find().isEmpty());
		} finally {
		  Files.delete(path);
		}
	}
	@Test @Tag("Q8")
	public void uriFinderFromPropertyFileThatDoesntExistShouldBeEmpty() throws IOException {
		var path = Files.createTempDirectory("healthcheck");
		var uriFinder = URIFinder.fromPropertyFile(path.resolve("this_file_doesnt_exist.txt"), "uri");
		assertTrue(uriFinder.find().isEmpty());
	}
	@Test @Tag("Q8")
	public void uriFinderFromPropertyFileWithTheWrongKeyShouldBeEmpty() throws IOException {
		var path = Files.createTempFile("", "healthcheck-empty.txt");
		Files.write(path, List.of("uri: http://www.google.fr", ""), UTF_8);
		try {
			var uriFinder = URIFinder.fromPropertyFile(path, "not_uri");
			assertTrue(uriFinder.find().isEmpty());
		} finally {
			Files.delete(path);
		}
	}
	@Test @Tag("Q8")
	public void uriFinderFromPropertyFileWithNull() {
		assertAll(
				() -> assertThrows(NullPointerException.class, () -> URIFinder.fromPropertyFile(null, "uri")),
				() -> assertThrows(NullPointerException.class, () -> URIFinder.fromPropertyFile(Path.of("nullcheck.txt"), null))
				);
	}
}
