open module fr.umlv.structconc {
  requires jdk.incubator.vector;

  requires jmh.core;   // annotations
  requires static jmh.generator.annprocess;  // only needed at compile time !
}