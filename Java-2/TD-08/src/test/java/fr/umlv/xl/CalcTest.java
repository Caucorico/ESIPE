package fr.umlv.xl;

import static java.util.stream.Collectors.toSet;
import static java.util.stream.IntStream.range;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

import fr.umlv.xl.Calc.Group;

@SuppressWarnings("static-method")
public class CalcTest {
  // Q1
  
  @Test
  public void testSetAndEval() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 2);
    assertEquals(2, (int)calc.eval("A1").orElse(-1));
  }

  @Test
  public void testEvalEmptyCell() {
    var calc = new Calc<Integer>();
    assertFalse(calc.eval("B2").isPresent());
  }
  
  @Test
  public void testSetTwice() {
    var calc = new Calc<Double>();
    calc.set("Z1", () -> 27.0);
    calc.set("Z1", () -> 42.0);
    assertTrue(42.0 == calc.eval("Z1").orElse(-1.0));
  }
  
  @Test
  public void testFunctionDependency() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 3);
    calc.set("B1", () -> 7 + calc.eval("A1").orElseThrow());
    assertEquals(10, (int)calc.eval("B1").orElse(-1));
  }
  
  @Test
  public void testSetCellNull() {
    var calc = new Calc<>();
    assertThrows(NullPointerException.class, () -> calc.set(null, () -> 2));
  }
  @Test
  public void testSetFunctionNull() {
    var calc = new Calc<>();
    assertThrows(NullPointerException.class, () -> calc.set("D9", null));
  }
  @Test
  public void testEvalNull() {
    var calc = new Calc<>();
    assertThrows(NullPointerException.class, () -> calc.eval(null));
  }
  
  
  // Q2
  
  @Test
  public void testToStringEmpty() {
    var calc = new Calc<Integer>();
    assertEquals("{}", calc.toString());
  }
  
  @Test
  public void testToStringOneCell() {
    var calc = new Calc<Integer>();
    calc.set("H3", () -> 777);
    assertEquals("{H3=777}", calc.toString());
  }
  @Test
  public void testToStringOneCell2() {
    var calc = new Calc<String>();
    calc.set("Z23", () -> "hello");
    assertEquals("{Z23=hello}", calc.toString());
  }

  @Test
  public void testToStringTwoCells() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 45);
    calc.set("C1", () -> 54);
    String s = calc.toString();
    assertTrue(s.equals("{A1=45, C1=54}") || s.equals("{A1=54, C1=45}"));
  }
  
  
  // Q3
  
  @Test
  public void testForEachEmpty() {
    var calc = new Calc<>();
    calc.forEach((cell, value) -> {
      fail("should not be called");
    });
  }

  @Test
  public void testForEachOneCell() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 666);
    var ok = new boolean[] { false };
    calc.forEach((cell, value) -> {
      ok[0] = true;
      assertEquals("A1", cell);
      assertEquals(666, (int)value);
    });
    assertTrue(ok[0]);
  }
  
  @Test
  public void testForEachSomeCells() {
    var calc = new Calc<String>();
    range(0, 10).forEach(i -> calc.set("Z" + i, () -> ""));
    
    var cells = new HashSet<String>();
    calc.forEach((cell, value) -> {
      cells.add(cell);
    });
    assertEquals(Set.of("Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9"), cells);
  }
  
  @Test
  public void testForEachALotOfCells() {
    var calc = new Calc<Integer>();
    range(0, 10_000).forEach(i -> calc.set("A" + i, () -> i));
    
    var cells = new HashSet<String>();
    calc.forEach((cell, value) -> {
      cells.add(cell);
    });
    assertEquals(range(0, 10_000).peek(i -> assertEquals(i, (int)calc.eval("A" + i).orElse(-1))).mapToObj(i -> "A" + i).collect(toSet()), cells);
  }
  
  @Test
  public void testForEachSignature() {
    var calc = new Calc<Integer>();
    calc.forEach((Object cell, Object value) -> {
      fail("should not be called");
    });
  }
  
  @Test
  public void testForEachNull() {
    var calc = new Calc<>();
    assertThrows(NullPointerException.class, () -> calc.forEach(null));
  }
  
  
  // Q4
  
  @Test
  public void testGroupOf() {
    var group = Group.of("foo", "bar");
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("foo", "bar"), set);
  }
  
  @Test
  public void testGroupCorrectlyTyped() {
    Group<String> group = Group.of("A1", "A2");
    assertNotNull(group);
  }
  
  @Test
  public void testGroupCorrectlyTypedEvenWithDifferentTypes() {
    Group<? extends Comparable<?>> group = Group.of("A1", 747);
    assertNotNull(group);
  }
  
  @Test
  public void testGroupOfEmpty() {
    var group = Group.of();
    group.values().forEach(value -> {
      fail("should not be called");
    });
  }
  
  @Test
  public void testGroupOfList() {
    var group = Group.of(List.of(1), List.of(2));
    group.values().forEach(list -> {
      assertEquals(1, list.size());
    });
  }
  
  @Test
  public void testGroupMutationAfterCreation() {
    var array = new String[] { "foo" };
    var group = Group.of(array);
    array[0] = "bar";
    assertEquals("foo", group.values().findFirst().orElseThrow());
  }
  
  @Test
  public void testGroupOfNull() {
    assertThrows(NullPointerException.class, () -> Group.of((Object[])null));
  }
  @Test
  public void testGroupOfNulls() {
    assertThrows(NullPointerException.class, () -> Group.of(null, null));
  }
  
  
  // Q5
  
  @Test
  public void testGroupForEach() {
    var group = Group.of("A1", "B2");
    var list = new ArrayList<String>();
    group.forEach(list::add);
    assertEquals(List.of("A1", "B2"), list);
  }
  
  @Test
  public void testGroupForEach2() {
    var group = Group.of(100, 88, 44, 67);
    var list = new ArrayList<Integer>();
    group.forEach(list::add);
    assertEquals(List.of(100, 88, 44, 67), list);
  }
  
  @Test
  public void testGroupForEachTypedCorrectly() {
    var group = Group.of("foo", "bar");
    group.forEach((Object o) -> {
      assertEquals(o, o.toString());
    });
  }
  
  @Test
  public void testGroupToListEmpty() {
    var group = Group.of();
    group.forEach(__ -> fail("should not be called"));
  }
  
  @Test
  public void testGroupForEachNull() {
    var group = Group.of("foo");
    assertThrows(NullPointerException.class, () -> group.forEach(null));
  }
  
  
  // Q6
  
  @Test
  public void testGroupCellMatrixOneCell() {
    var group = Group.cellMatrix(1, 1, 'A', 'A');
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("A1"), set);
  }
  
  @Test
  public void testGroupCellMatrixCorrectlyTyped() {
    var group = Group.cellMatrix(1, 1, 'A', 'A');
    assertNotNull(group);
  }
  
  @Test
  public void testGroupCellMatrixSeveralCells() {
    var group = Group.cellMatrix(1, 2, 'A', 'B');
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("A1", "A2", "B1", "B2"), set);
  }
  
  @Test
  public void testGroupCellMatrixSameRow() {
    var group = Group.cellMatrix(1, 1, 'A', 'B');
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("A1", "B1"), set);
  }
  
  @Test
  public void testGroupCellMatrixSameColumn() {
    var group = Group.cellMatrix(1, 2, 'A', 'A');
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("A1", "A2"), set);
  }
  
  @Test
  public void testGroupCellMatrixBadOrderRow() {
    assertThrows(IllegalArgumentException.class, () -> Group.cellMatrix(2, 1, 'A', 'A'));
  }
  @Test
  public void testGroupCellMatrixBadOrderColumn() {
    assertThrows(IllegalArgumentException.class, () -> Group.cellMatrix(1, 1, 'B', 'A'));
  }
  
  
  // Q7
  
  @Test
  public void testGroupIgnore() {
    var group = Group.of("A1", "B2", "C3").ignore(Set.of("B2"));
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("A1", "C3"), set);
  }
  
  @Test
  public void testGroupIgnoreEmpty() {
    var group = Group.of("X1", "Y2", "Z3").ignore(Set.of());
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("X1", "Y2", "Z3"), set);
  }
  
  @Test
  public void testGroupIgnoreNonExisting() {
    var group = Group.of("X99", "K2000").ignore(Set.of("X99", "Z4"));
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("K2000"), set);
  }
  
  @Test
  public void testGroupIgnoreCellMatrix() {
    var group = Group.cellMatrix(7, 9, 'A', 'C').ignore(Set.of("B8"));
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("A7", "A8", "A9", "B7", "B9", "C7", "C8", "C9"), set);
  }
  
  @Test
  public void testGroupIgnoreSignature() {
    var empty = Set.of();
    var group = Group.of("A1").ignore(empty);
    var set = new HashSet<String>();
    group.values().forEach(set::add);
    assertEquals(Set.of("A1"), set);
  }
  
  @Test
  public void testGroupIgnoreNull() {
    assertThrows(NullPointerException.class, () -> Group.of("A1").ignore(null));
  }
  
  
  // Q8
  
  @Test
  public void testGroupEval() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 1);
    calc.set("B2", () -> 10);
    assertEquals(11, Group.of("A1","B2").eval(calc::eval).mapToInt(x -> x).sum());
  }
  
  @Test
  public void testGroupEval2() {
    var calc = new Calc<Double>();
    calc.set("A1", () -> 2.0);
    calc.set("B2", () -> 4.0);
    assertTrue(6.0 == Group.of("A1","B2").eval(calc::eval).mapToDouble(x -> x).sum());
  }
  
  @Test
  public void testGroupEvalWithCellWithoutValue() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 2);
    calc.set("B2", () -> 20);
    assertEquals(22, Group.of("A1","B2", "C3").eval(calc::eval).mapToInt(x -> x).sum());
  }
  
  @Test
  public void testGroupEvalCellMatrix() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 3);
    calc.set("B2", () -> 30);
    assertEquals(33, Group.cellMatrix(1, 2, 'A', 'B').eval(calc::eval).mapToInt(x -> x).sum());
  }
  
  @Test
  public void testGroupEvalCellMatrixCount() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 2);
    calc.set("Z99", () -> 20);
    assertEquals(2, Group.cellMatrix(1, 99, 'A', 'Z').eval(calc::eval).count());
  }
  
  @Test
  public void testGroupEvalCellMatrixIgnore() {
    var calc = new Calc<Integer>();
    calc.set("A1", () -> 4);
    calc.set("B2", () -> 40);
    calc.set("C3", () -> 400);
    assertEquals(404, Group.cellMatrix(1, 3, 'A', 'C').ignore(Set.of("B2")).eval(calc::eval).mapToInt(x -> x).sum());
  }
  
  @Test
  public void testGroupEvalNull() {
    assertThrows(NullPointerException.class, () -> Group.of("A1").eval(null));
  }
}
